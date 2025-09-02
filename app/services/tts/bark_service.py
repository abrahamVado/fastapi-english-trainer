# app/services/tts/bark_service.py
from __future__ import annotations
import os

# ---- HARD DISABLE CUDA FOR BARK/TORCH (must be set before torch/bark imports) ----
# Hide all CUDA devices so Torch reports 0 GPUs (and Bark won't poke CUDA at import)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Bark smaller models by default
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "1")

# Optional: reduce Torch CUDA allocator fragmentation if someone later enables CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

# ---- Import torch and monkey-patch cuda probes to be safe on systems with broken drivers ----
import torch  # noqa: E402

# Some Torch builds still think CUDA is "available" even with 0 visible devices.
# Make sure any probe Bark does returns CPU-only semantics.
try:
    import types  # noqa: E402
    _cuda = torch.cuda

    def _false(*args, **kwargs): return False
    def _zero(*args, **kwargs): return 0

    # Only patch if there are zero visible devices (or if you want to force CPU unconditionally)
    if _cuda.device_count() == 0:
        _cuda.is_available = _false          # type: ignore[attr-defined]
        _cuda.is_initialized = _false        # type: ignore[attr-defined]
        _cuda.device_count = _zero           # type: ignore[attr-defined]
        # Newer torch adds this API; ensure it doesn't touch device 0
        if hasattr(_cuda, "is_bf16_supported"):
            _cuda.is_bf16_supported = _false   # type: ignore[attr-defined]
except Exception:
    # If anything goes wrong here, we prefer failing "closed" (CPU only)
    pass

# ---- Now it is safe to import Bark; it will see CPU only ----
import io, re, wave, asyncio, inspect, logging, gc  # noqa: E402
from typing import Optional  # noqa: E402
import numpy as np  # noqa: E402

try:
    from bark import generate_audio, preload_models  # noqa: E402
    from bark.generation import SAMPLE_RATE  # noqa: E402
except Exception:
    from bark.api import generate_audio          # type: ignore  # noqa: E402
    from bark.generation import SAMPLE_RATE, preload_models  # type: ignore  # noqa: E402

log = logging.getLogger("tts.bark")


class BarkService:
    """
    Bark TTS wrapper (CPU + small models).
    """

    _is_warmed_small = False
    _is_warmed_full = False
    _sem = asyncio.Semaphore(1)

    def __init__(
        self,
        voice_preset: str = "v2/es_speaker_0",
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        use_small_default: bool = True,
    ):
        self.voice_preset = voice_preset
        self.text_temp = float(text_temp)
        self.waveform_temp = float(waveform_temp)
        self.use_small_default = bool(use_small_default)

    @classmethod
    def _supported_kwargs(cls):
        try:
            import inspect as _inspect
            sig = _inspect.signature(preload_models)
            return set(sig.parameters.keys())
        except Exception:
            return set()

    def _preload(self, *, use_small: bool):
        supported = self._supported_kwargs()
        desired = {
            "use_small": use_small,
            "text_use_small": use_small,
            "coarse_use_small": use_small,
            "fine_use_small": use_small,
            "codec_use_small": use_small,
            # Some Bark builds accept device=; on CPU-only this is harmless to omit
            # "device": "cpu",
        }
        kwargs = {k: v for k, v in desired.items() if k in supported}
        log.warning("[Bark] preload_models small=%s kwargs=%s", use_small, list(kwargs.keys()))
        try:
            if kwargs:
                preload_models(**kwargs)
            else:
                preload_models()
        except TypeError:
            preload_models()

    def warm_models(self, *, use_small: bool = True):
        if use_small and self._is_warmed_small:
            return
        if (not use_small) and self._is_warmed_full:
            return
        # Ensure env for older releases
        os.environ["SUNO_USE_SMALL_MODELS"] = "1" if use_small else "0"
        self._preload(use_small=use_small)
        if use_small:
            self._is_warmed_small = True
        else:
            self._is_warmed_full = True

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
        s = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        pcm = (s * 32767.0).astype(np.int16)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        bio.seek(0)
        return bio.getvalue()

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[\.\?\!;])\s+|,\s{1,3}", text)
        chunks, buf = [], ""
        for p in parts:
            nxt = (buf + " " + p).strip() if buf else p
            if len(nxt) <= 120:
                buf = nxt
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return chunks[:12]

    async def synthesize_bytes(
        self,
        text: str,
        *,
        voice: str | None = None,
        use_small: bool | None = None,
        seed: int | None = None,
        text_temp: float | None = None,
        waveform_temp: float | None = None,
    ) -> bytes:
        small = self.use_small_default if use_small is None else bool(use_small)
        # Warm models
        try:
            self.warm_models(use_small=small)
        except Exception as e:
            log.error("[Bark] warm_models failed (CPU-only): %s", e, exc_info=True)
            # reset state just in case; then try again once
            self._is_warmed_small = self._is_warmed_full = False
            gc.collect()
            self.warm_models(use_small=small)

        preset = (voice or self.voice_preset) or "v2/es_speaker_0"
        tt = self.text_temp if text_temp is None else float(text_temp)
        wt = self.waveform_temp if waveform_temp is None else float(waveform_temp)

        chunks = self._split_into_chunks(text) or [""]

        async with self._sem:
            def _run_once():
                audio_all = None
                for ch in chunks:
                    audio = generate_audio(
                        ch,
                        history_prompt=preset,
                        text_temp=tt,
                        waveform_temp=wt,
                    )
                    audio_all = audio if audio_all is None else np.concatenate([audio_all, audio])
                return audio_all if audio_all is not None else np.zeros(1, dtype=np.float32)

            loop = asyncio.get_running_loop()
            samples = await loop.run_in_executor(None, _run_once)

        return self._to_wav_bytes(samples)

    async def synthesize_stream(self, *args, **kwargs):
        wav = await self.synthesize_bytes(*args, **kwargs)
        yield wav


# Singleton accessor
_service: Optional[BarkService] = None
def get_bark_service() -> BarkService:
    global _service
    if _service is None:
        _service = BarkService(use_small_default=True)
    return _service

def reset_bark_service():
    global _service
    _service = BarkService(use_small_default=True)
    gc.collect()
