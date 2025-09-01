# app/api/routers/tts.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import io, wave, math, struct, numpy as np
from typing import Tuple, Union
import logging

log = logging.getLogger("tts")
router = APIRouter(prefix="/tts", tags=["tts"])

def _ensure_array(audio: Union[np.ndarray, list]) -> np.ndarray:
    if audio is None:
        return np.zeros((0,), dtype=np.float32)
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    else:
        audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio

def _to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1, dtype=np.float32) if audio.shape[1] else np.zeros((0,), np.float32)
    return audio.reshape(-1).astype(np.float32, copy=False)

def float_to_pcm16_wav(audio: Union[np.ndarray, list], sr: int = 22050) -> io.BytesIO:
    audio = _ensure_array(audio)
    audio = _to_mono_f32(audio)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16, copy=False)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf

def gen_beep_wav(duration_s: float = 0.4, freq: float = 880.0, sr: int = 16000) -> io.BytesIO:
    n = int(duration_s * sr)
    buf = io.BytesIO()
    def w(fmt, *v): buf.write(struct.pack(fmt, *v))
    buf.write(b"RIFF"); w("<I", 36 + n*2); buf.write(b"WAVEfmt ")
    w("<I", 16); w("<H", 1); w("<H", 1); w("<I", sr); w("<I", sr*2); w("<H", 2); w("<H", 16)
    buf.write(b"data"); w("<I", n*2)
    for i in range(n):
        s = int(32767 * math.sin(2*math.pi*freq*i/sr))
        w("<h", s)
    buf.seek(0)
    return buf

async def _run_synth(text: str) -> Tuple[np.ndarray, int]:
    # TODO: change this import to your real TTS implementation
    # from app.services.tts.my_tts import synth
    from app.services.tts.my_tts import synth  # <-- verify this path exists
    result = synth(text)
    if hasattr(result, "__await__"):  # coroutine?
        audio, sr = await result
    else:
        audio, sr = result
    return audio, int(sr)

@router.post("/say")
async def tts_say(req: dict):
    text = (req.get("text") or "").strip()
    if not text:
        text = "Hello"
    log.warning(f"[TTS] request text='{text[:64]}'")

    try:
        audio, sr = await _run_synth(text)
        audio = _ensure_array(audio)
        if audio.size == 0:
            log.error("[TTS] synth returned empty audio")
            wav = gen_beep_wav()
            return StreamingResponse(
                wav, media_type="audio/wav",
                headers={"Cache-Control": "no-store", "X-TTS-Empty": "1"}
            )
        wav = float_to_pcm16_wav(audio, sr)
        size = wav.getbuffer().nbytes
        log.warning(f"[TTS] ok sr={sr} samples={audio.size} bytes={size}")
        return StreamingResponse(wav, media_type="audio/wav",
                                 headers={"Cache-Control": "no-store",
                                          "X-TTS-SR": str(sr),
                                          "X-TTS-Samples": str(audio.size)})
    except Exception as e:
        log.exception("[TTS] synth failed")
        wav = gen_beep_wav()
        return StreamingResponse(
            wav, media_type="audio/wav",
            headers={"Cache-Control": "no-store", "X-TTS-Error": type(e).__name__}
        )

@router.post("/warm")
async def tts_warm():
    return {"ok": True}
