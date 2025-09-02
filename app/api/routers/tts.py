# app/api/routers/tts.py
from __future__ import annotations

import io
import logging
import math
import struct
import wave
from typing import Tuple, Union, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# NEW: use your Ollama client for tutor replies
from app.services.judge.ollama_client import OllamaJudge

log = logging.getLogger("tts")
router = APIRouter(prefix="/tts", tags=["tts"])

# single client is fine (httpx inside handles sessions)
_judge = OllamaJudge()

# -------------------- Request model --------------------

class SpeakRequest(BaseModel):
    # If use_llm is False -> this is the text to synth directly.
    # If use_llm is True  -> this is the "user said" given to the tutor LLM.
    text: Optional[str] = Field(None, max_length=1500, description="Text (or seed)")

    # Turn LLM->TTS pipeline on
    use_llm: bool = Field(False, description="If true, call Ollama to craft a reply, then TTS it.")

    # Optional context to steer the tutor (only used when use_llm=True)
    role: Optional[str] = Field(None, description="Learner role (e.g. 'backend developer')")
    level: Optional[str] = Field(None, description="Proficiency (e.g. 'intermediate')")
    mode: Optional[str]  = Field(None, description="Scenario (e.g. 'interview')")
    llm_model: Optional[str] = Field(None, description="Override OLLAMA_MODEL just for this call")

    # If text is omitted you can pull the user's latest answer from /sim
    session_id: Optional[str] = None
    question_id: Optional[str] = None

    # Reserved knobs (wire up later to Bark config if you want)
    voice: Optional[str] = Field(None, description="Voice/persona id")
    speed: Optional[float] = Field(None, gt=0.25, lt=4.0, description="Playback speed multiplier")

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

def float_to_pcm16_wav(audio: Union[np.ndarray, list, None], sr: int) -> io.BytesIO:
    audio = _normalize_peak(_to_mono_f32(_ensure_array(audio)))
    pcm16 = (audio * 32767.0).astype(np.int16, copy=False)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf

def gen_beep_wav(duration_s: float = 0.35, freq: float = 880.0, sr: int = 24000) -> io.BytesIO:
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
    buf.seek(0)
    return buf

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
    """Fetch the last transcribed answer from the in-memory /sim store."""
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

@router.post("/say", response_class=StreamingResponse, summary="Speak provided text or tutor LLM reply")
async def tts_say(req: SpeakRequest):
    source = "TEXT"
    tts_text = (req.text or "").strip()

    if req.use_llm:
        # If no explicit text, try pulling from /sim store using ids
        if not tts_text and req.session_id and req.question_id:
            tts_text = _pull_sim_answer(req.session_id, req.question_id).strip()
        if not tts_text:
            raise HTTPException(status_code=422, detail="No input text for LLM. Provide 'text' or (session_id, question_id).")

        log.info("[TTS] LLM mode: model=%s role=%s level=%s mode=%s",
                 req.llm_model or "(default)", req.role, req.level, req.mode)

        reply = await _judge.a_tutor_reply(
            user_text=tts_text,
            role=req.role or "",
            level=req.level or "",
            mode=req.mode or "",
            model=req.llm_model,      # None → env default (OLLAMA_MODEL)
            max_tokens=100,
            temperature=0.3,
        )
        tts_text = (reply or "").strip()
        source = "LLM"

        if not tts_text:
            wav = gen_beep_wav()
            return StreamingResponse(
                wav, media_type="audio/wav",
                headers={"Cache-Control": "no-store", "X-TTS-Empty": "1", "X-TTS-Source": source},
            )

    if not tts_text:
        # fall back to a friendly phrase
        tts_text = "Hello"
        source = "DEFAULT"

    log.info("[TTS] /say len=%d source=%s preview=%r", len(tts_text), source, tts_text[:64])

    try:
        audio, sr = await _run_synth(tts_text)
        if audio.size == 0:
            log.error("[TTS] Backend returned empty audio (sr=%s). Beep fallback.", sr)
            wav = gen_beep_wav(sr=max(sr, 8000))
            return StreamingResponse(
                wav, media_type="audio/wav",
                headers={"Cache-Control": "no-store", "X-TTS-Empty": "1", "X-TTS-Source": source},
            )

        wav = float_to_pcm16_wav(audio, sr)
        size = wav.getbuffer().nbytes
        headers = {
            "Cache-Control": "no-store",
            "X-TTS-SR": str(sr),
            "X-TTS-Samples": str(audio.size),
            "X-TTS-Source": source,                 # TEXT | LLM | DEFAULT
            "X-LLM-Model": req.llm_model or "",     # useful for debugging
            "Content-Length": str(size),
        }
        return StreamingResponse(wav, media_type="audio/wav", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("[TTS] synth failed: %s", e)
        wav = gen_beep_wav()
        return StreamingResponse(
            wav, media_type="audio/wav",
            headers={"Cache-Control": "no-store", "X-TTS-Error": type(e).__name__, "X-TTS-Source": source},
            status_code=500,
        )

@router.post("/warm", summary="Preload models / warm the TTS backend")
async def tts_warm():
    try:
        # instantiate Bark once if you want to reduce first-call latency
        from app.services.tts.bark_tts import _get_service  # optional helper if you exposed it
        _get_service()
        return {"ok": True, "warmed": True}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}
