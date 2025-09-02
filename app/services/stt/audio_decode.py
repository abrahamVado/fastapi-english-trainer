# app/services/stt/audio_decode.py
from __future__ import annotations
import io
import numpy as np

def decode_to_float32_mono_16k(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Use PyAV (ffmpeg bindings) to decode *any* typical container to
    float32 mono @16k. Falls back to soundfile if PyAV is missing.
    """
    try:
        import av  # type: ignore
        with av.open(io.BytesIO(raw_bytes), mode="r") as container:
            astream = next(s for s in container.streams if s.type == "audio")
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
            chunks = []
            for frame in container.decode(astream):
                frame = resampler.resample(frame)
                for p in frame.planes:
                    chunks.append(p.to_bytes())
            pcm16 = b"".join(chunks)
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 16000
    except Exception:
        # Fallback: try soundfile (handles wav, flac, ogg, m4a, mp3 depending on libsndfile)
        try:
            import soundfile as sf  # type: ignore
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
            mono = data.mean(axis=1).astype("float32", copy=False)
            if sr != 16000:
                # simple resample via librosa if available; otherwise leave sr as-is
                try:
                    import librosa  # type: ignore
                    mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
                    sr = 16000
                except Exception:
                    pass
            return mono, sr
        except Exception as e:
            raise RuntimeError(f"Audio decode failed: {type(e).__name__}: {e}")
