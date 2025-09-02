# app/api/routers/tts.py
from __future__ import annotations

import io
import logging
import math
import struct
import wave
import re
import time
from pathlib import Path
from typing import Tuple, Union, Optional, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.tts.bark_service import get_bark_service
from app.services.judge.ollama_client import OllamaJudge

log = logging.getLogger("tts")
router = APIRouter(prefix="/tts", tags=["tts"])

# --- Lazy singleton for Ollama judge (prevents NameError/import-order issues)
_judge_singleton: Optional[OllamaJudge] = None

def _get_judge() -> OllamaJudge:
    global _judge_singleton
    if _judge_singleton is None:
        _judge_singleton = OllamaJudge()
    return _judge_singleton


# -------------------- Idempotency (in-memory) --------------------
# If you run multiple workers/hosts, swap this for Redis.
_REQ_CACHE: dict[str, tuple[int, bytes]] = {}
_REQ_TTL_MS = 5 * 60 * 1000

def _now_ms() -> int:
    return int(time.time() * 1000)

def _gc_cache() -> None:
    now = _now_ms()
    for k in list(_REQ_CACHE.keys()):
        if now - _REQ_CACHE[k][0] > _REQ_TTL_MS:
            _REQ_CACHE.pop(k, None)

def _cache_get(req_id: Optional[str]) -> Optional[bytes]:
    if not req_id:
        return None
    hit = _REQ_CACHE.get(req_id)
    return hit[1] if hit else None

def _cache_put(req_id: Optional[str], wav: bytes) -> None:
    if not req_id:
        return
    _REQ_CACHE[req_id] = (_now_ms(), wav)

# -------------------- Voices --------------------
VOICE_DIR = Path(__file__).resolve().parents[2] / "app" / "voices"
VOICE_PATH = VOICE_DIR / "es_0.npz"

BUILTIN_VOICES = {
    "en": [f"v2/en_speaker_{i}" for i in range(10)],
    "es": [f"v2/es_speaker_{i}" for i in range(5)],
    "fr": [f"v2/fr_speaker_{i}" for i in range(4)],
    "de": [f"v2/de_speaker_{i}" for i in range(4)],
    "hi": [f"v2/hi_speaker_{i}" for i in range(2)],
    "zh": ["v2/zh_speaker_0"],
    "ja": ["v2/ja_speaker_0"],
}

def _list_local_npz() -> list[str]:
    try:
        if not VOICE_DIR.exists():
            return []
        return sorted(str(p) for p in VOICE_DIR.glob("*.npz"))
    except Exception:
        log.exception("[TTS] listing local voices failed")
        return []

# -------------------- Length helpers --------------------
def _word_budget(req) -> tuple[int, str, int]:
    """
    Returns (max_words, style_hint_for_llm, max_tokens_for_llm).
    Defaults to SHORT answers unless caller overrides.
    """
    if getattr(req, "max_words", None):
        mw = max(5, int(req.max_words))
    elif getattr(req, "target_seconds", None):
        mw = max(5, int(float(req.target_seconds) * 2.5))  # ~2.5 words/sec
    else:
        length = (getattr(req, "length", None) or "short").lower()
        mw = {"short": 20, "medium": 45, "long": 100}.get(length, 20)

    if mw <= 20:
        hint = f"Reply in ≤{mw} words. Use one simple sentence."
    elif mw <= 45:
        hint = f"Reply in ≤{mw} words. Use up to two short sentences."
    else:
        hint = f"Reply in ≤{mw} words. Use a brief, compact paragraph."

    mt = getattr(req, "max_tokens", None) or int(mw * 1.4)
    return mw, hint, mt

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
def _clamp_words(text: str, max_words: int) -> str:
    text = (text or "").strip()
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    parts = _SENT_SPLIT.split(text)
    kept, count = [], 0
    for p in parts:
        w = len(p.split())
        if count + w > max_words:
            break
        kept.append(p)
        count += w
    out = " ".join(kept).strip() if kept else " ".join(words[:max_words]).strip()
    if out and out[-1] not in ".?!":
        out += "…"
    return out

def _dedupe_tts_text(txt: str, user_seed: str = "") -> str:
    """
    Remove sentence repeats, short n-gram loops, and user-echo from TTS text.
    """
    s = (txt or "").strip()
    if not s:
        return s

    # If the LLM echoed the user's text, strip it.
    if user_seed:
        seed = user_seed.strip()
        if len(seed) > 6 and seed.lower() in s.lower():
            s = re.sub(re.escape(seed), "", s, flags=re.IGNORECASE).strip()

    # Sentence-level dedupe
    parts = _SENT_SPLIT.split(s)  # split on end punctuation
    out = []
    for p in parts:
        if not p:
            continue
        p_norm = p.strip()
        if out and p_norm == out[-1].strip():
            continue  # drop identical consecutive sentence
        # Short n-gram loops like "in the in the"
        p2 = re.sub(r"\b((\w+\s+){1,3}\w+)\b(?:\s+\1\b){1,}", r"\1", p_norm, flags=re.IGNORECASE)
        # Single-word loops like "really really"
        p3 = re.sub(r"(\b[\w\'\-]{3,}\b)(?:\s+\1){1,}", r"\1", p2, flags=re.IGNORECASE)
        out.append(p3)

    s2 = " ".join(out).strip()
    return re.sub(r"\s{2,}", " ", s2).strip()


# -------------------- Request models --------------------
class SpeakRequest(BaseModel):
    # If use_llm is False -> synth directly. If True -> tutor LLM crafts reply then TTS.
    text: Optional[str] = Field(None, max_length=1500, description="Text (or seed)")
    use_llm: bool = Field(False, description="If true, call Ollama to craft a reply, then TTS it.")

    # Tutor context (only when use_llm=True)
    role: Optional[str]  = Field(None, description="Learner role (e.g. 'backend developer')")
    level: Optional[str] = Field(None, description="Proficiency (e.g. 'intermediate')")
    mode: Optional[str]  = Field(None, description="Scenario (e.g. 'interview')")
    llm_model: Optional[str] = Field(None, description="Override OLLAMA_MODEL just for this call")

    # If text omitted, you can pull user's latest answer from /sim
    session_id: Optional[str] = None
    question_id: Optional[str] = None

    # Voice & playback knobs
    voice: Optional[str] = Field(None, description="Voice/persona id or .npz prompt path")
    speed: Optional[float] = Field(None, gt=0.25, lt=4.0, description="Playback speed multiplier")

    # Length controls
    length: Optional[Literal["short", "medium", "long"]] = Field(None, description="High-level length hint")
    max_words: Optional[int] = Field(None, description="Hard cap on words (overrides length)")
    target_seconds: Optional[int] = Field(None, description="Approx duration → word budget (~2.5 w/s)")
    max_tokens: Optional[int] = Field(None, description="Forwarded to LLM; overrides default token budget")

class PreviewRequest(BaseModel):
    voice: Optional[str] = Field(None, description="Voice id or .npz path")
    text: Optional[str] = Field(None, description="Preview text")
    speed: Optional[float] = Field(None, gt=0.25, lt=4.0)

# -------------------- Audio helpers --------------------
def _ensure_array(audio: Union[np.ndarray, list, None]) -> np.ndarray:
    if audio is None:
        return np.zeros((0,), dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

def _to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[1] in (1, 2):
            return audio.mean(axis=1, dtype=np.float32)
        if audio.shape[0] in (1, 2):
            return audio.mean(axis=0, dtype=np.float32)
    return audio.reshape(-1).astype(np.float32, copy=False)

def _normalize_peak(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    if audio.size == 0:
        return audio
    m = float(np.max(np.abs(audio)))
    if m > 0:
        audio = (audio / m) * peak
    return np.clip(audio, -1.0, 1.0)

def float_to_pcm16_wav_bytes(audio: Union[np.ndarray, list, None], sr: int) -> bytes:
    audio = _normalize_peak(_to_mono_f32(_ensure_array(audio)))
    pcm16 = (audio * 32767.0).astype(np.int16, copy=False)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

def gen_beep_wav_bytes(duration_s: float = 0.35, freq: float = 880.0, sr: int = 24000) -> bytes:
    n = int(duration_s * sr)
    buf = io.BytesIO()
    def w(fmt, *v): buf.write(struct.pack(fmt, *v))
    buf.write(b"RIFF"); w("<I", 36 + n*2); buf.write(b"WAVEfmt ")
    w("<I", 16); w("<H", 1); w("<H", 1); w("<I", sr); w("<I", sr*2); w("<H", 2); w("<H", 16)
    buf.write(b"data"); w("<I", n*2)
    twopi_over_sr = 2.0 * math.pi / sr
    for i in range(n):
        s = int(32767 * math.sin(twopi_over_sr * freq * i))
        w("<h", s)
    return buf.getvalue()

# -------------------- TTS glue --------------------
async def _run_synth(text: str) -> Tuple[np.ndarray, int]:
    from app.services.tts.bark_tts import synth  # async or sync
    result = synth(text)
    if hasattr(result, "__await__"):
        audio, sr = await result
    else:
        audio, sr = result
    return _ensure_array(audio), int(sr)

def _pull_sim_answer(session_id: Optional[str], question_id: Optional[str]) -> str:
    if not session_id or not question_id:
        return ""
    try:
        from app.api.routers import sim
        turns = sim._TURNS.get(session_id)
        if not turns:
            return ""
        t = next((t for t in turns if t.get("qid") == question_id), None)
        return (t.get("answer_text") or "") if t else ""
    except Exception:
        return ""

# -------------------- Routes --------------------
@router.get("/say")
def tts_ping():
    return {"ok": True, "route": "/tts/say"}

@router.post("/say", summary="Speak provided text or tutor LLM reply")
async def tts_say(
    req: SpeakRequest,
    x_req_id: Optional[str] = Header(None, convert_underscores=False),
):
    """
    One-shot WAV bytes (not chunked streaming).
    Idempotent: same X-Req-Id -> same WAV without re-synth.
    """
    _gc_cache()

    # Idempotency: return cached WAV if present
    cached = _cache_get(x_req_id)
    if cached is not None:
        return Response(
            cached,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Idempotency": "hit",
                "Content-Length": str(len(cached)),
            },
        )

    source = "TEXT"
    tts_text = (req.text or "").strip()

    # Decide budgets up-front
    max_words, style_hint, max_tok = _word_budget(req)

    # LLM mode
    if req.use_llm:
        if not tts_text and req.session_id and req.question_id:
            tts_text = _pull_sim_answer(req.session_id, req.question_id).strip()
        if not tts_text:
            raise HTTPException(
                status_code=422,
                detail="No input text for LLM. Provide 'text' or (session_id, question_id)."
            )
        log.info("[TTS] LLM mode: model=%s role=%s level=%s mode=%s",
                 req.llm_model or "(default)", req.role, req.level, req.mode)

        seed_for_deecho = tts_text  # preserve the user's text to suppress echo

        user_for_llm = (
            f"{seed_for_deecho}\n\n"
            f"[Instruction: {style_hint} "
            f"Do NOT repeat my words. Do NOT quote or restate the user. "
            f"Give only your reply—no preambles.]"
        )

        judge = _get_judge()
        reply = await judge.a_tutor_reply(
            user_text=user_for_llm,
            role=req.role or "",
            level=req.level or "",
            mode=req.mode or "",
            model=req.llm_model,
            max_tokens=max_tok,
            temperature=0.3,
        )

        tts_text = _dedupe_tts_text((reply or "").strip(), user_seed=seed_for_deecho)
        source = "LLM"


    if not tts_text:
        tts_text, source = "Hello", "DEFAULT"

    # Enforce max words server-side
    tts_text = _clamp_words(_dedupe_tts_text(tts_text), max_words)
    debug_preview = tts_text[:120]

    log.info("[TTS] /say len=%d (max_words=%d) source=%s preview=%r",
             len(tts_text), max_words, source, tts_text[:64])

    # ---- Try Bark first ----
    try:
        svc = get_bark_service()
        voice_arg = str(VOICE_PATH) if VOICE_PATH.exists() else (req.voice or "v2/en_speaker_7")

        wav_bytes = await svc.synthesize_bytes(
            tts_text,
            voice=voice_arg,
            seed=12345,
        )

        # Optional playback speed
        if req.speed and abs(req.speed - 1.0) > 1e-3:
            try:
                import soundfile as sf, librosa
                y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y2 = librosa.effects.time_stretch(y, rate=float(req.speed))
                wav_bytes = float_to_pcm16_wav_bytes(y2, sr)
            except Exception as e:
                log.warning("[TTS] speed stretch failed (%s); returning original audio", e)

        _cache_put(x_req_id, wav_bytes)
        return Response(
            wav_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Idempotency": "store" if x_req_id else "none",
                "X-TTS-SR": "24000",
                "X-TTS-Source": source,
                "X-LLM-Model": req.llm_model or "",
                "Content-Length": str(len(wav_bytes)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("[TTS] Bark synth failed: %s — falling back", e)

    # ---- Fallback synth ----
    try:
        audio, sr = await _run_synth(tts_text)
        if audio.size == 0:
            log.error("[TTS] Backend returned empty audio (sr=%s). Beep fallback.", sr)
            wav_bytes = gen_beep_wav_bytes(sr=max(sr, 8000))
            _cache_put(x_req_id, wav_bytes)
            return Response(
                wav_bytes, media_type="audio/wav",
                headers={
                    "Cache-Control": "no-store",
                    "X-Idempotency": "store" if x_req_id else "none",
                    "X-TTS-Empty": "1",
                    "X-TTS-Source": source,
                    "Content-Length": str(len(wav_bytes)),
                }
            )

        wav_bytes = float_to_pcm16_wav_bytes(audio, sr)
        _cache_put(x_req_id, wav_bytes)
        return Response(
            wav_bytes, media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Idempotency": "store" if x_req_id else "none",
                "X-TTS-SR": str(sr),
                "X-TTS-Samples": str(audio.size),
                "X-TTS-Source": source,
                "X-LLM-Model": req.llm_model or "",
                "Content-Length": str(len(wav_bytes)),
            },
        )

    except Exception as e2:
        log.exception("[TTS] fallback synth failed: %s", e2)
        wav_bytes = gen_beep_wav_bytes()
        # Even errors get cached for the same req id to stay idempotent
        _cache_put(x_req_id, wav_bytes)
        return Response(
            wav_bytes, media_type="audio/wav", status_code=500,
            headers={
                "Cache-Control": "no-store",
                "X-Idempotency": "store" if x_req_id else "none",
                "X-TTS-Error": type(e2).__name__,
                "X-TTS-Source": "ERROR",
                "Content-Length": str(len(wav_bytes)),
            },
        )

@router.post("/warm", summary="Preload models / warm the TTS backend")
async def tts_warm():
    try:
        from app.services.tts.bark_tts import _get_service  # if exposed
        _get_service()
        return {"ok": True, "warmed": True}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}

@router.get("/voices", summary="List available voices (built-ins + local .npz)")
def tts_voices():
    return {
        "builtin": BUILTIN_VOICES,
        "local": _list_local_npz(),
        "default": "v2/es_speaker_0",
    }

@router.post("/preview", summary="Render a short sample for a given voice")
async def tts_preview(req: PreviewRequest):
    preview_text = (req.text or "Hola, esta es una muestra de voz breve.").strip()
    svc = get_bark_service()
    voice_arg = req.voice or "v2/es_speaker_0"

    try:
        wav_bytes = await svc.synthesize_bytes(preview_text, voice=voice_arg, seed=12345)
        return Response(
            wav_bytes, media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-TTS-Voice": voice_arg,
                "Content-Length": str(len(wav_bytes)),
            },
        )
    except Exception as e:
        log.exception("[TTS] preview failed: %s", e)
        return Response(
            gen_beep_wav_bytes(), media_type="audio/wav", status_code=500,
            headers={"Cache-Control": "no-store", "X-TTS-Error": type(e).__name__},
        )
