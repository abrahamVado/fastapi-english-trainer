# app/api/routers/sim.py
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

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

# --- Services (singletons) ---------------------------------------------------
_asr = WhisperService(settings.WHISPER_MODEL)
_judge = OllamaJudge()  # async/sync capable per the improved client

# --- Simple in-memory stores -------------------------------------------------
# NOTE: in-memory stores reset on process restart and are not multi-worker safe.
_SESS: Dict[str, Dict] = {}         # session_id -> {"role":..,"level":..,"mode":..}
_TURNS: Dict[str, List[Dict]] = {}  # session_id -> [{qid, q, answer_text, ...}, ...]

# --- LLM rubric --------------------------------------------------------------
_CONTENT_RUBRIC = """Grade content from 0-100:
- Correctness (40)
- Depth (30)
- Clarity (20)
- Examples (10)
Return strict JSON: {"score":int,"key_points":[...],"gaps":[...],"tips":[...]}.
"""


def _ask_first_question(role: str, level: str, mode: str) -> str:
    # TODO: later: route to LLM interviewer based on role/level/mode
    return f"Tell me about a challenge you solved using {role} at {level} level."


def _ask_followup(prev_answer: str) -> str:
    # TODO: can be made smarter using prev_answer and LLM
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
    audio: UploadFile = File(...)
):
    turns = _TURNS.get(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="session not found")
    t = next((t for t in turns if t["qid"] == question_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="question not found")

    data = await audio.read()
    # Optional: pass initial_prompt or language hints to ASR in the future
    asr_text = await _asr.transcribe(data, language="en")
    t["answer_text"] = asr_text or ""
    return SimAnswerAudioRes(
        session_id=session_id,
        question_id=question_id,
        asr_text=asr_text or "",
        confidence=None,  # TODO: add from WhisperService if available
    )


@router.post("/score", response_model=SimScoreRes)
async def sim_score(req: SimScoreReq):
    """
    Scores the last answer for the given question:
    - Content (via Ollama judge, async)
    - Pronunciation (via IPA mapping, if expected_text provided)
    - Fluency (simple heuristic)
    """
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
        # fail-soft so the rest of scoring continues
        judge_json = {"score": 0, "tips": [f"(ollama error: {type(e).__name__})"]}

    try:
        content = int(judge_json.get("score", 0))
    except Exception:
        content = 0

    # 2) PRONUNCIATION
    if req.expected_text:
        # Reading-mode strict scoring
        try:
            pron_json = score_pronunciation(req.expected_text, ans or "")
            pronunciation = int(pron_json["overall"].get("score_0_100", 0))
        except Exception:
            pronunciation = 0
    else:
        # Open-answer fallback until you score audio directly
        pronunciation = 80 if ans else 0

    # 3) FLUENCY (cheap heuristic for now)
    w = len(ans.split())
    fluency = 70 + min(20, w // 20) if ans else 0

    overall = round(0.55 * content + 0.30 * pronunciation + 0.15 * fluency)

    # persist scores in turn
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
