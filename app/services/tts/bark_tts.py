# app/services/tts/bark_tts.py
from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional, Tuple

import numpy as np
from .bark_service import BarkService

log = logging.getLogger("tts.bark")

# ------------------ Env defaults (can be overridden before boot) ------------------
# Smaller Bark models unless explicitly disabled:
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "1")
# Prefer CPU for Bark to avoid GPU contention with Whisper:
os.environ.setdefault("TTS_BARK_DEVICE", "cpu")  # set to "cuda" to try GPU first

# ------------------ Service lifecycle ------------------
_SVC: Optional[BarkService] = None

def _get_service() -> BarkService:
    """Lazy-create a BarkService instance (respects env vars at creation)."""
    global _SVC
    if _SVC is None:
        _SVC = BarkService()
        log.warning("[Bark] BarkService initialized (small=%s, device_pref=%s)",
                    os.environ.get("SUNO_USE_SMALL_MODELS"),
                    os.environ.get("TTS_BARK_DEVICE"))
    return _SVC

def _reset_service():
    """Drop the cached service so next call recreates it with current env."""
    global _SVC
    _SVC = None
    log.warning("[Bark] BarkService reset (will reinitialize on next call)")

# ------------------ Audio helpers ------------------

def _pcm16_bytes_to_float32(frames: bytes) -> np.ndarray:
    """PCM16 little-endian bytes -> float32 array in [-1, 1]."""
    if not frames:
        return np.zeros((0,), dtype=np.float32)
    a = np.frombuffer(frames, dtype=np.int16, count=len(frames) // 2)
    # /32768.0 handles -32768 correctly; final clip guards any rounding edge
    a = (a.astype(np.float32) / 32768.0)
    return np.clip(a, -1.0, 1.0)

def _to_mono_f32(audio: np.ndarray, channels: int) -> np.ndarray:
    """Ensure mono float32. If stereo (interleaved), average L/R."""
    if audio.size == 0:
        return audio
    if channels == 1:
        return audio.astype(np.float32, copy=False)
    if channels == 2:
        return audio.reshape(-1, 2).mean(axis=1).astype(np.float32, copy=False)
    # Unexpected channel count: flatten as a safe fallback
    return audio.astype(np.float32, copy=False)

def _normalize_peak(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Peak-normalize to `peak` then clip to [-1, 1]."""
    if audio.size == 0:
        return audio
    m = float(np.max(np.abs(audio)))
    if m > 0.0:
        audio = (audio / m) * peak
    return np.clip(audio, -1.0, 1.0)

def _wbytes_to_mono_f32_sr(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes (PCM16) to mono float32 and return (audio, sr)."""
    if not wav_bytes:
        return np.zeros((0,), dtype=np.float32), 24000
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nch = wf.getnchannels()
        sw  = wf.getsampwidth()
        sr  = int(wf.getframerate()) or 24000
        n   = wf.getnframes()
        comp= wf.getcomptype()
        if comp != "NONE":
            log.error("[Bark] Unsupported WAV compression: %s", comp)
            return np.zeros((0,), dtype=np.float32), sr
        if sw != 2:  # expect PCM16
            log.error("[Bark] Unexpected sample width: %d (want 16-bit)", sw * 8)
            return np.zeros((0,), dtype=np.float32), sr
        frames = wf.readframes(n)
    audio = _pcm16_bytes_to_float32(frames)
    audio = _to_mono_f32(audio, nch)
    audio = _normalize_peak(audio, peak=0.99)
    return audio, sr

# ------------------ Public API ------------------

async def synth(text: str) -> Tuple[np.ndarray, int]:
    """
    Generate speech audio from text using BarkService.
    Always returns (audio_float32_mono_in_-1..1, sample_rate_int).
    On any error, returns (empty_array, sr) so the caller can beep-fallback.
    """
    if not text or not text.strip():
        return np.zeros((0,), dtype=np.float32), 24000  # Bark default SR

    # Honor device preference on first attempt
    device_pref = (os.environ.get("TTS_BARK_DEVICE") or "").strip().lower()
    try:
        if device_pref == "cpu":
            # Ensure CPU-only before service (and model) creation
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        wav_bytes = await _get_service().synthesize_bytes(text)
        audio, sr = _wbytes_to_mono_f32_sr(wav_bytes)
        return audio, sr

    except Exception as e:
        msg = str(e).lower()
        gpu_related = ("out of memory" in msg) or ("cuda" in msg) or ("cudnn" in msg)

        # If first attempt may have used GPU (or failed GPU alloc), retry once on CPU with small models.
        if device_pref != "cpu" or gpu_related:
            try:
                log.error("[Bark] synth failed on GPU (%s). Retrying on CPU (small models).", e)
                # Force CPU and small models; reset service so next call reloads with these envs.
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["SUNO_USE_SMALL_MODELS"] = "1"
                os.environ["TTS_BARK_DEVICE"] = "cpu"
                _reset_service()

                wav_bytes = await _get_service().synthesize_bytes(text)
                audio, sr = _wbytes_to_mono_f32_sr(wav_bytes)
                return audio, sr
            except Exception as e2:
                log.exception("[Bark] CPU fallback failed: %s", e2)
                return np.zeros((0,), dtype=np.float32), 24000

        # Non-GPU-related failure: log and return empty
        log.exception("[Bark] synth failed: %s", e)
        return np.zeros((0,), dtype=np.float32), 24000
