# app/services/tts/bark_service.py
from __future__ import annotations
import os, io, re, wave, asyncio, inspect, logging, gc
from typing import Optional

import numpy as np
import torch

# ---- Env knobs --------------------------------------------------------------
# Expose GPU 0 by default (override via real env if needed). If you want CPU-only, set CUDA_VISIBLE_DEVICES=-1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SUNO_USE_SMALL_MODELS", "1")  # safer default

# Import Bark after env is set
try:
    from bark import generate_audio, preload_models
    from bark.generation import SAMPLE_RATE
except Exception:
    from bark.api import generate_audio          # type: ignore
    from bark.generation import SAMPLE_RATE, preload_models  # type: ignore

log = logging.getLogger("tts.bark")


class BarkService:
    """
    Bark TTS wrapper.
    - prefer_cuda: opt into GPU if available; else CPU.
    - use_small_default: load the 'small' variants (recommended even on GPU).
    - voice consistency: one seed per utterance + single history_prompt for all chunks.
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
        prefer_cuda: bool = False,
    ):
        self.voice_preset = voice_preset
        self.text_temp = float(text_temp)
        self.waveform_temp = float(waveform_temp)
        self.use_small_default = bool(use_small_default)

        self._prefer_cuda = bool(prefer_cuda) and torch.cuda.is_available()
        self._device = torch.device("cuda") if self._prefer_cuda else torch.device("cpu")

    @classmethod
    def _supported_kwargs(cls):
        try:
            sig = inspect.signature(preload_models)
            return set(sig.parameters.keys())
        except Exception:
            return set()

    def _preload(self, *, use_small: bool):
        supported = self._supported_kwargs()
        desired = {
            "use_small": use_small,           # some builds
            "text_use_small": use_small,      # others
            "coarse_use_small": use_small,
            "fine_use_small": use_small,
            "codec_use_small": use_small,
        }
        kwargs = {k: v for k, v in desired.items() if k in supported}
        if "device" in supported:
            kwargs["device"] = self._device
        log.warning("[Bark] preload_models small=%s device=%s kwargs=%s",
                    use_small, self._device, list(kwargs.keys()))
        try:
            preload_models(**kwargs) if kwargs else preload_models()
        except TypeError:
            preload_models()

    def warm_models(self, *, use_small: bool = True):
        # pick device each warm in case availability changed
        self._device = torch.device("cuda") if (self._prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")
        os.environ["SUNO_USE_SMALL_MODELS"] = "1" if use_small else "0"

        if use_small and self._is_warmed_small:
            return
        if (not use_small) and self._is_warmed_full:
            return

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

    def _seed_all(self, seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    async def synthesize_bytes(
        self,
        text: str,
        *,
        voice: str | None = None,
        use_small: bool | None = None,
        seed: int | None = 12345,
        text_temp: float | None = None,
        waveform_temp: float | None = None,
    ) -> bytes:
        small = self.use_small_default if use_small is None else bool(use_small)
        try:
            self.warm_models(use_small=small)
        except Exception as e:
            log.error("[Bark] warm_models failed: %s", e, exc_info=True)
            self._is_warmed_small = self._is_warmed_full = False
            gc.collect()
            self.warm_models(use_small=small)

        if seed is not None:
            self._seed_all(int(seed))

        preset = (voice or self.voice_preset) or "v2/es_speaker_0"
        hp = preset
        try:
            if isinstance(preset, str) and preset.lower().endswith(".npz") and os.path.exists(preset):
                from bark.generation import load_history_prompt
                hp = load_history_prompt(preset)
        except Exception:
            hp = preset

        tt = (self.text_temp if text_temp is None else float(text_temp)) or 0.5
        wt = (self.waveform_temp if waveform_temp is None else float(waveform_temp)) or 0.5

        chunks = self._split_into_chunks(text) or [""]

        SINGLE_CALL_CHAR_LIMIT = 220
        if len(text.strip()) <= SINGLE_CALL_CHAR_LIMIT:
            async with self._sem:
                def _one():
                    return generate_audio(text.strip(), history_prompt=hp, text_temp=tt, waveform_temp=wt)
                loop = asyncio.get_running_loop()
                samples = await loop.run_in_executor(None, _one)
            return self._to_wav_bytes(samples)

        async with self._sem:
            def _run_once():
                audio_all = None
                for ch in chunks:
                    audio = generate_audio(ch, history_prompt=hp, text_temp=tt, waveform_temp=wt)
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
        # prefer_cuda=True tries GPU; falls back to CPU if unavailable
        _service = BarkService(use_small_default=True, prefer_cuda=True)
    return _service

def reset_bark_service():
    global _service
    _service = BarkService(use_small_default=True, prefer_cuda=True)
    gc.collect()
