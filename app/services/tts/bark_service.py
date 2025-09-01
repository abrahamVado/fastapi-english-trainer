# app/services/tts/bark_service.py
from __future__ import annotations
import io, re, wave, os, asyncio
import numpy as np
import inspect 


# Bark 0.1.5 can expose APIs in slightly different places; make this robust.
try:
    from bark import generate_audio, preload_models
    from bark.generation import SAMPLE_RATE
except Exception:
    from bark.api import generate_audio          # type: ignore
    from bark.generation import SAMPLE_RATE, preload_models  # type: ignore

import torch

# Prefer CUDA if present, but don’t crash if missing
if torch.cuda.is_available():
    try:
        torch.set_default_device("cuda")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass  # safe fallback

class BarkService:
    """
    Bark TTS wrapper.
    - Generates 24 kHz audio from text using a voice preset (history_prompt).
    - By default returns a single WAV as bytes (fits <audio>).
    - Concurrency is limited via a semaphore to protect GPU VRAM.
    """

    _is_warmed_full = False
    _is_warmed_small = False
    _sem = asyncio.Semaphore(1)   # Bark is heavy; raise cautiously if VRAM allows.

    def __init__(
        self,
        voice_preset: str = "v2/es_speaker_0",
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        use_small_default: bool = False,
    ):
        self.voice_preset = voice_preset
        self.text_temp = float(text_temp)
        self.waveform_temp = float(waveform_temp)
        self.use_small_default = bool(use_small_default)

        # default to full models unless SMALL explicitly requested
        os.environ.setdefault("SUNO_USE_SMALL_MODELS", "false")

    @classmethod
    def warm_models(cls, *, use_small: bool = False):
        """
        Version-tolerant warmup:
        - If this Bark build accepts per-stage small flags, pass only the ones it supports.
        - If it only accepts `use_small`, pass that.
        - If it accepts no kwargs, call with no args.
        - Also toggles SUNO_USE_SMALL_MODELS for older releases.
        """
        # avoid duplicate warmups
        if use_small and cls._is_warmed_small:
            return
        if (not use_small) and cls._is_warmed_full:
            return

        # some Bark builds respect this env var only
        os.environ["SUNO_USE_SMALL_MODELS"] = "true" if use_small else "false"

        # figure out what preload_models actually accepts
        try:
            sig = inspect.signature(preload_models)
            supported = set(sig.parameters.keys())
        except Exception:
            supported = set()

        # candidate kwargs across versions
        desired_kwargs = {
            "use_small": use_small,                 # some versions
            "text_use_small": use_small,            # others
            "coarse_use_small": use_small,
            "fine_use_small": use_small,
            "codec_use_small": use_small,           # may be missing (causing your error)
        }

        # keep only supported kwargs
        filtered = {k: v for k, v in desired_kwargs.items() if k in supported}

        # try the most specific first, then fall back
        try:
            if filtered:
                preload_models(**filtered)
            else:
                preload_models()
        except TypeError:
            # signature mismatch → final fallback
            preload_models()

        if use_small:
            cls._is_warmed_small = True
        else:
            cls._is_warmed_full = True


    @staticmethod
    def _to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
        # Bark returns float32 [-1, 1]. Convert to 16-bit PCM WAV.
        s = np.clip(samples, -1.0, 1.0)
        pcm = (s * 32767.0).astype(np.int16)

        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return bio.getvalue()

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        # Lightweight sentence-ish splitter to keep latency reasonable.
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[\.\?\!;])\s+|,\s{1,3}", text)

        chunks, buf = [], ""
        for p in parts:
            nxt = (buf + " " + p).strip() if buf else p
            if len(nxt) < 100:  # keep chunks under ~100 chars
                buf = nxt
            else:
                if buf: chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return chunks

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
        """
        Generate a single WAV (bytes). Safe to return via Response/StreamingResponse.
        """
        use_small = self.use_small_default if use_small is None else use_small
        self.warm_models(use_small=use_small)

        preset = (voice or self.voice_preset) or "v2/es_speaker_0"
        tt = self.text_temp if text_temp is None else float(text_temp)
        wt = self.waveform_temp if waveform_temp is None else float(waveform_temp)

        chunks = self._split_into_chunks(text) or [""]

        # Serialize heavy GPU work
        async with self._sem:
            audio_all = None
            for ch in chunks:
                audio = generate_audio(
                    ch,
                    history_prompt=preset,
                    text_temp=tt,
                    waveform_temp=wt
                )
                audio_all = audio if audio_all is None else np.concatenate([audio_all, audio])

        return self._to_wav_bytes(audio_all if audio_all is not None else np.zeros(1, dtype=np.float32))

    async def synthesize_stream(self, *args, **kwargs):
        """
        Async generator that yields a *single* WAV blob.
        (Keeping generator signature so you can later emit per-sentence chunks if you switch to MP3/WebM streaming.)
        """
        wav = await self.synthesize_bytes(*args, **kwargs)
        yield wav
