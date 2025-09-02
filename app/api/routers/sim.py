# app/api/routers/sim.py
from __future__ import annotations

import uuid, hashlib, time, logging, re
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.sim import (
    SimStartReq, SimStartRes, SimNextReq, SimNextRes,
    SimAnswerTextReq, SimAnswerAudioRes, SimScoreReq, SimScoreRes,
    ScoreBreakdown, SimReportRes
)
from app.services.stt.whisper_service import WhisperService
from app.services.judge.ollama_client import OllamaJudge
from app.services.ipa.mapping import score_pronunciation

router = APIRouter(prefix="/sim", tags=["sim"])
log = logging.getLogger("sim")

# --- Services (singletons) ---------------------------------------------------
_asr = WhisperService(settings.WHISPER_MODEL)
_judge = OllamaJudge()  # async/sync capable per the improved client

# --- Simple in-memory stores -------------------------------------------------
# NOTE: in-memory stores reset on process restart and are not multi-worker safe.
_SESS: Dict[str, Dict] = {}         # session_id -> {"role":..,"level":..,"mode":..}
_TURNS: Dict[str, List[Dict]] = {}  # session_id -> [{qid, q, answer_text, ...}, ...]

# --- Idempotency & duplicate detection (server-side "eco" killers) ----------
# 1) Response cache by client request id (works even if audio changes)
_REQ_CACHE: Dict[str, Tuple[int, dict]] = {}   # X-Req-Id -> (ts_ms, result_json)
_REQ_TTL_MS = 5 * 60 * 1000

# 2) Existing blob-idempotency cache (your original) keyed by (client_key + blob hash)
_IDEMP: Dict[str, Tuple[int, dict]] = {}       # key -> (ts_ms, result_json)
_IDEMP_TTL_MS = 5 * 60 * 1000

# 3) Duplicate audio fingerprint per (session_id, question_id)
_AUDIO_FP: Dict[str, Tuple[int, str]] = {}     # "sid:qid" -> (ts_ms, sha1_hex)
_AUDIO_TTL_MS = 15 * 60 * 1000

def _now_ms() -> int:
    return int(time.time() * 1000)

def _gc_maps():
    now = _now_ms()
    # clean request-id cache
    for k in list(_REQ_CACHE.keys()):
        if now - _REQ_CACHE[k][0] > _REQ_TTL_MS:
            _REQ_CACHE.pop(k, None)
    # clean idempotency cache
    for k in list(_IDEMP.keys()):
        if now - _IDEMP[k][0] > _IDEMP_TTL_MS:
            _IDEMP.pop(k, None)
    # clean audio fingerprints
    for k in list(_AUDIO_FP.keys()):
        if now - _AUDIO_FP[k][0] > _AUDIO_TTL_MS:
            _AUDIO_FP.pop(k, None)

def _idem_key(client_key: Optional[str], audio_bytes: bytes) -> str:
    h = hashlib.sha1(audio_bytes).hexdigest()
    return f"{client_key or 'noid'}:{h}"

def _audio_key(session_id: str, question_id: str) -> str:
    return f"{session_id}:{question_id}"

def _fingerprint(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# --- Strong repetition cleaner for ASR text ----------------------------------
import difflib

_CLAUSE_SPLIT = re.compile(r"[\.!\?,;:\n]\s*")

def _norm_clause(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # common “are you” → “nargus” style confusions still compare well after norm,
    # but we also defang hello-loops by trimming repeated hello tokens later.
    return s

def _similar(a: str, b: str) -> float:
    # Fast path for substrings
    if not a or not b:
        return 0.0
    if a in b or b in a:
        # Treat substring as highly similar
        la, lb = len(a), len(b)
        return min(la, lb) / max(la, lb)
    return difflib.SequenceMatcher(None, a, b).ratio()

def _dedupe_text(asr_text: str) -> str:
    s = (asr_text or "").strip()
    if not s:
        return s

    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT.split(s) if c and c.strip()]
    if not raw_clauses:
        return s

    kept: list[str] = []
    kept_norm: list[str] = []

    for c in raw_clauses:
        # kill tiny token loops inside a clause (e.g., "really really")
        c2 = re.sub(r"\b((\w+\s+){1,3}\w+)\b(?:\s+\1\b){1,}", r"\1", c, flags=re.IGNORECASE)
        c2 = re.sub(r"(\b[\w\'\-]{3,}\b)(?:\s+\1){1,}", r"\1", c2, flags=re.IGNORECASE)
        cn = _norm_clause(c2)
        if not cn:
            continue

        # drop very short fragments that look like restarts ("hello my friend", "are you")
        words = cn.split()
        is_short = len(words) <= 4

        drop = False
        for i, prev in enumerate(kept_norm):
            sim = _similar(cn, prev)
            # If it's a short fragment and largely overlaps an earlier clause, drop it
            if is_short and sim >= 0.75:
                drop = True
                break
            # If it's a near-duplicate even when not short, keep only the longer/clearer one
            if sim >= 0.88:
                # prefer the longer raw text
                if len(c2) <= len(kept[i]):
                    drop = True
                else:
                    kept[i] = c2
                    kept_norm[i] = cn
                break

        if drop:
            continue

        kept.append(c2)
        kept_norm.append(cn)

    # collapse repetitive "hello" lead-ins across kept clauses
    for i in range(len(kept)):
        kept[i] = re.sub(r"^(hello[\s,;:!-]+){1,}", "Hello, ", kept[i], flags=re.IGNORECASE).strip()
        # trim duplicated "hello" between clauses by normalizing again later

    # Rejoin into a clean sentence or two; remove consecutive exact duplicates
    out: list[str] = []
    for c in kept:
        if out and _norm_clause(c) == _norm_clause(out[-1]):
            continue
        out.append(c)

    final = ". ".join(out).strip()
    # tidy punctuation
    final = re.sub(r"\s+([?!.,;:])", r"\1", final)
    if not final.endswith((".", "!", "?")):
        final += "."
    return final


# --- LLM rubric --------------------------------------------------------------
_CONTENT_RUBRIC = """Grade content from 0-100:
- Correctness (40)
- Depth (30)
- Clarity (20)
- Examples (10)
Return strict JSON: {"score":int,"key_points":[...],"gaps":[...],"tips":[...]}.
"""

def _ask_first_question(role: str, level: str, mode: str) -> str:
    return f"Tell me about a challenge you solved using {role} at {level} level."

def _ask_followup(prev_answer: str) -> str:
    return "Thanks. Can you explain the trade-offs you considered?"

# --- Endpoints ---------------------------------------------------------------

@router.post("/start", response_model=SimStartRes)
def sim_start(req: SimStartReq):
    sid = uuid.uuid4().hex
    _SESS[sid] = req.model_dump()
    qid = uuid.uuid4().hex
    q = _ask_first_question(req.role, req.level, req.mode)
    _TURNS[sid] = [{"qid": qid, "q": q, "answer_text": ""}]
    return SimStartRes(session_id=sid, question_id=qid, question=q)

@router.post("/next", response_model=SimNextRes)
def sim_next(req: SimNextReq):
    if req.session_id not in _SESS:
        raise HTTPException(status_code=404, detail="session not found")
    last = _TURNS[req.session_id][-1]
    qid = uuid.uuid4().hex
    q = _ask_followup(last.get("answer_text", ""))
    _TURNS[req.session_id].append({"qid": qid, "q": q, "answer_text": ""})
    return SimNextRes(question_id=qid, question=q)

@router.post("/answer/text")
def sim_answer_text(req: SimAnswerTextReq):
    turns = _TURNS.get(req.session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="session not found")
    t = next((t for t in turns if t["qid"] == req.question_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="question not found")
    t["answer_text"] = (req.text or "").strip()
    return {"ok": True}

@router.post("/answer/audio", response_model=SimAnswerAudioRes)
async def sim_answer_audio(
    session_id: str = Form(...),
    question_id: str = Form(...),
    audio: UploadFile = File(...),
    client_audio_id: Optional[str] = Form(None),
    x_req_id: Optional[str] = Header(None, convert_underscores=False),  # <— client sends the same id to TTS
):
    # validate session/turn
    turns = _TURNS.get(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="session not found")
    t = next((t for t in turns if t["qid"] == question_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="question not found")

    _gc_maps()

    # --- (A) Return cached result if same X-Req-Id (client retried) ----------
    if x_req_id and x_req_id in _REQ_CACHE:
        cached = _REQ_CACHE[x_req_id][1]
        # keep turn state consistent
        t["answer_text"] = cached.get("asr_text", "") or ""
        return JSONResponse(cached)

    # --- (B) Read audio & compute keys ---------------------------------------
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio upload")

    idem = _idem_key(x_req_id or client_audio_id, data)

    # --- (C) If same blob already processed anywhere, return cached ----------
    if idem in _IDEMP:
        cached = _IDEMP[idem][1]
        t["answer_text"] = cached.get("asr_text", "") or ""
        # also map X-Req-Id cache for this turn
        if x_req_id:
            _REQ_CACHE[x_req_id] = (_now_ms(), cached)
        return JSONResponse(cached)

    # --- (D) If same bytes posted again for same (session,question), ignore ---
    akey = _audio_key(session_id, question_id)
    data_fp = _fingerprint(data)
    prev = _AUDIO_FP.get(akey)
    if prev and prev[1] == data_fp:
        res = {
            "session_id": session_id,
            "question_id": question_id,
            "asr_text": "",
            "confidence": None,
            "note": "duplicate audio ignored",
        }
        # cache under all maps so subsequent retries stay consistent
        _IDEMP[idem] = (_now_ms(), res)
        if x_req_id:
            _REQ_CACHE[x_req_id] = (_now_ms(), res)
        return JSONResponse(res)

    # record latest fp for this (session, question)
    _AUDIO_FP[akey] = (_now_ms(), data_fp)

    # --- (E) Build domain hint for ASR ---------------------------------------
    sess = _SESS.get(session_id, {})
    initial_prompt = (
        f"Interview context. Role: {sess.get('role','')}. "
        f"Level: {sess.get('level','')}. Mode: {sess.get('mode','')}. "
        "Common tech terms: Kubernetes, gRPC, PostgreSQL, TypeScript, React, Kafka, Terraform, AWS."
    )

    # --- (F) Transcribe with your service; then lightly de-repeat ------------
    asr_text = await _asr.transcribe(
        data,
        language="en",
        initial_prompt=initial_prompt,
        content_type=(audio.content_type or "")
    )
    asr_text = _dedupe_text(asr_text or "")

    # --- (G) Persist + cache + log -------------------------------------------
    t["answer_text"] = asr_text
    result = {
        "session_id": session_id,
        "question_id": question_id,
        "asr_text": asr_text,
        "confidence": None,
    }
    now = _now_ms()
    _IDEMP[idem] = (now, result)
    if x_req_id:
        _REQ_CACHE[x_req_id] = (now, result)

    short_hash = data_fp[:10]
    log.info("[ASR] sid=%s qid=%s bytes=%d hash=%s text_len=%d text=%r",
             session_id, question_id, len(data), short_hash, len(asr_text), asr_text[:140])

    return JSONResponse(result)

@router.post("/score", response_model=SimScoreRes)
async def sim_score(req: SimScoreReq):
    turns = _TURNS.get(req.session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="session not found")
    t = next((t for t in turns if t["qid"] == req.question_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="question not found")

    ans = (t.get("answer_text") or "").strip()

    # 1) CONTENT via Ollama judge (async)
    sess = _SESS.get(req.session_id, {})
    evidence = {
        "question": t.get("q", ""),
        "answer_text": ans,
        "role": sess.get("role", ""),
        "level": sess.get("level", ""),
        "mode": sess.get("mode", "")
    }

    try:
        judge_json = await _judge.judge_content(_CONTENT_RUBRIC, evidence)
    except Exception as e:
        judge_json = {"score": 0, "tips": [f"(ollama error: {type(e).__name__})"]}

    try:
        content = int(judge_json.get("score", 0))
    except Exception:
        content = 0

    # 2) PRONUNCIATION
    if req.expected_text:
        try:
            pron_json = score_pronunciation(req.expected_text, ans or "")
            pronunciation = int(pron_json["overall"].get("score_0_100", 0))
        except Exception:
            pronunciation = 0
    else:
        pronunciation = 80 if ans else 0

    # 3) FLUENCY (cheap heuristic for now)
    w = len(ans.split())
    fluency = 70 + min(20, w // 20) if ans else 0

    overall = round(0.55 * content + 0.30 * pronunciation + 0.15 * fluency)

    t["content_score"] = content
    t["pron_score"] = pronunciation
    t["fluency_score"] = fluency
    t["overall_score"] = overall

    tips = judge_json.get("tips") or [
        "Add a concrete example.",
        "Explain trade-offs and impact.",
        "Keep answers structured (STAR).",
    ]

    return SimScoreRes(
        scores=ScoreBreakdown(
            content=content,
            pronunciation=pronunciation,
            fluency=fluency,
            overall=overall,
        ),
        tips=tips,
    )

@router.get("/report", response_model=SimReportRes)
def sim_report(session_id: str):
    turns = _TURNS.get(session_id, [])
    scored = [t.get("overall_score") for t in turns if t.get("overall_score") is not None]
    avg = round(sum(scored) / max(1, len(scored))) if scored else 0

    return SimReportRes(
        session_id=session_id,
        turns=[{
            "qid": t["qid"],
            "q": t["q"],
            "answer_text": t.get("answer_text", ""),
            "scores": {
                "content": t.get("content_score", 0),
                "pronunciation": t.get("pron_score", 0),
                "fluency": t.get("fluency_score", 0),
                "overall": t.get("overall_score", 0),
            }
        } for t in turns],
        overall_avg=avg
    )

@router.post("/answer/llm")
async def sim_answer_llm(session_id: str, question_id: str):
    turns = _TURNS.get(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="session not found")
    t = next((t for t in turns if t["qid"] == question_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="question not found")

    sess = _SESS.get(session_id, {})
    role  = sess.get("role", "")
    level = sess.get("level", "")
    mode  = sess.get("mode", "")

    user_text = (t.get("answer_text") or "").strip()
    if not user_text:
        user_text = f"(No answer captured yet.) The question was: {t.get('q','')}"

    reply = await _judge.a_tutor_reply(
        user_text,
        role=role,
        level=level,
        mode=mode,
        max_tokens=256,
        temperature=0.6,
    )
    t["tutor_reply"] = reply
    return {"reply": reply}
