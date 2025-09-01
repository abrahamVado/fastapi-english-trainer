import io, wave, numpy as np
from .bark_service import BarkService

async def synth(text: str):
    """
    Uses BarkService to synthesize WAV bytes, then returns (float32 mono array, sample_rate).
    Router can await this (it's async).
    """
    svc = BarkService()
    wav_bytes = await svc.synthesize_bytes(text)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        frames = wf.readframes(n)

    # PCM16 -> float32 [-1..1]
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, int(sr)