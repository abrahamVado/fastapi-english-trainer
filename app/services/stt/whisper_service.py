# app/services/stt/whisper_service.py
from __future__ import annotations
from typing import Optional
import os, tempfile, asyncio, logging

import torch
from faster_whisper import WhisperModel

log = logging.getLogger("whisper")
_lock = asyncio.Lock()  # prevent concurrent loads

class WhisperService:
    """
    Faster-Whisper wrapper with:
      - CUDA -> CPU fallback on OOM
      - Quantized defaults for small VRAM
      - Singleton model instance
      - Env-configurable knobs
    """
    _model: Optional[WhisperModel] = None
    _loaded_cfg: Optional[tuple] = None  # (model_name, device, compute_type, device_index)

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        compute_type: str = None,
        device_index: int = None,
    ):
        # Defaults via env (safe for low VRAM)
        self.model_name   = model_name   or os.getenv("WHISPER_MODEL", "base.en")  # try "tiny.en" if still tight
        self.device       = device       or os.getenv("WHISPER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE") or ("int8_float16" if self.device == "cuda" else "int8")
        self.device_index = device_index if device_index is not None else int(os.getenv("WHISPER_DEVICE_INDEX", "0"))

        log.warning(f"[WhisperService] target model='{self.model_name}' device={self.device} "
                    f"compute={self.compute_type} index={self.device_index}")

    async def _load_model(self, force_cpu=False):
        """
        Load (or reuse) the singleton model. If force_cpu=True, load on CPU even if device was CUDA.
        """
        target_device = "cpu" if force_cpu else self.device
        target_compute = self.compute_type
        if force_cpu:
            # smallest CPU footprint
            target_compute = "int8"

        cfg = (self.model_name, target_device, target_compute, self.device_index)

        if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
            return

        async with _lock:
            # double-check inside lock
            if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
                return
            log.warning(f"[WhisperService] loading Whisper: model={cfg[0]} device={cfg[1]} compute={cfg[2]} idx={cfg[3]}")
            WhisperService._model = WhisperModel(
                self.model_name,
                device=target_device,
                device_index=self.device_index,
                compute_type=target_compute,
            )
            WhisperService._loaded_cfg = cfg

    async def _ensure_model(self):
        # First try as configured (likely CUDA+int8_float16)
        try:
            await self._load_model(force_cpu=False)
        except RuntimeError as e:
            msg = str(e).lower()
            # Typical faster-whisper / ctranslate2 OOM message
            if "out of memory" in msg or "cuda" in msg and "failed" in msg:
                log.error("[WhisperService] CUDA OOM; falling back to CPU quantized model")
                await self._load_model(force_cpu=True)
            else:
                raise
    # app/services/stt/whisper_service.py (patch)
    async def transcribe(self, wav_bytes: bytes, language: str = "en",
                        initial_prompt: Optional[str] = None,
                        content_type: str = "audio/webm") -> str:
        await self._ensure_model()

        # guess extension from content-type (fallback to .wav)
        ct = (content_type or "").lower()
        if "webm" in ct:
            suffix = ".webm"
        elif "ogg" in ct or "opus" in ct:
            suffix = ".ogg"
        elif "wav" in ct:
            suffix = ".wav"
        elif "m4a" in ct or "mp4" in ct or "aac" in ct:
            suffix = ".m4a"
        elif "mpeg" in ct or "mp3" in ct:
            suffix = ".mp3"
        else:
            suffix = ".wav"  # safe default if you send real WAV

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name

        try:
            segments, info = WhisperService._model.transcribe(
                tmp,
                language=language,
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
                best_of=int(os.getenv("WHISPER_BEST_OF", "1")),
                temperature=float(os.getenv("WHISPER_TEMPERATURE", "0")),
                initial_prompt=(initial_prompt[:1800] if initial_prompt else None),
                word_timestamps=False,
            )
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text
        finally:
            try: os.remove(tmp)
            except: pass
