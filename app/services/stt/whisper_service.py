# app/services/stt/whisper_service.py
from __future__ import annotations
from typing import Optional, Tuple
import os, tempfile, asyncio, logging
import re
from difflib import SequenceMatcher

import torch
from faster_whisper import WhisperModel

log = logging.getLogger("whisper")
_lock = asyncio.Lock()  # prevent concurrent model loads

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v

def _is_cuda_alloc_error(msg: str) -> bool:
    m = msg.lower()
    return (
        "cudnn_status" in m
        or "device_allocation_failed" in m
        or "out of memory" in m
        or ("cuda" in m and "failed" in m)
    )


def _normalize_for_compare(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.?!]+$", "", s)     # strip trailing punctuation for compare
    return s

def _dedupe_sentences(text: str) -> str:
    """
    Remove immediate duplicate sentences/phrases and collapse long repeats.
    """
    # Split on sentences (simple heuristic)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    last_norm = ""
    for p in parts:
        p_norm = _normalize_for_compare(p)
        # exact duplicate?
        if p_norm and p_norm == last_norm:
            continue
        # near-duplicate? (e.g., repeated by chunk stitching)
        if p_norm and last_norm and SequenceMatcher(None, p_norm, last_norm).ratio() >= 0.92:
            continue
        out.append(p)
        last_norm = p_norm

    # collapse tripled words: "I I I" -> "I", "uh uh uh" -> "uh"
    s = " ".join(out)
    s = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", s, flags=re.IGNORECASE)
    # reduce multiple spaces, final tidy
    s = re.sub(r"\s+", " ", s).strip()
    return s


class WhisperService:
    """
    Faster-Whisper wrapper with:
      - CUDA -> CPU fallback on OOM / cuDNN errors
      - Low-VRAM defaults
      - Singleton model instance across requests
      - Env-configurable knobs

    ENV (all optional):
      WHISPER_MODEL           (default: "tiny.en")
      WHISPER_DEVICE          (default: "cuda" if available else "cpu")
      WHISPER_DEVICE_INDEX    (default: "0")
      WHISPER_COMPUTE         (default: "int8_float16" on cuda, "int8" on cpu)
      WHISPER_BEAM_SIZE       (default: "1")
      WHISPER_BEST_OF         (default: "1")
      WHISPER_TEMPERATURE     (default: "0")
      WHISPER_CHUNK_LENGTH    (default: "15")   # seconds
      WHISPER_ENC_BATCH       (default: "4")    # encoder batch size
    """
    _model: Optional[WhisperModel] = None
    _loaded_cfg: Optional[Tuple[str, str, str, int]] = None  # (model_name, device, compute_type, device_index)

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        device_index: Optional[int] = None,
    ):
        # Safer default: tiny.en (fast + light). You can bump to base.en if GPU is roomy.
        self.model_name   = model_name   or _env("WHISPER_MODEL", "tiny.en")
        self.device       = device       or _env("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = int(device_index if device_index is not None else _env("WHISPER_DEVICE_INDEX", "0"))
        # compute type default depends on device
        self.compute_type = compute_type or _env("WHISPER_COMPUTE", "int8_float16" if self.device == "cuda" else "int8")

        # Transcription knobs (low VRAM defaults)
        self.beam_size     = int(_env("WHISPER_BEAM_SIZE", "1"))
        self.best_of       = int(_env("WHISPER_BEST_OF", "1"))
        self.temperature   = float(_env("WHISPER_TEMPERATURE", "0"))
        self.chunk_length  = int(_env("WHISPER_CHUNK_LENGTH", "15"))
        #self.enc_batch     = int(_env("WHISPER_ENC_BATCH", "4"))

        log.warning(
            "[WhisperService] target model='%s' device=%s compute=%s index=%d "
            "(beam=%d best_of=%d temp=%.2f chunk=%ds)",
            self.model_name, self.device, self.compute_type, self.device_index,
            self.beam_size, self.best_of, self.temperature, self.chunk_length
        )


    async def _load_model(self, device: Optional[str] = None, compute_type: Optional[str] = None):
        """
        Load (or reuse) the singleton model with the requested device/compute.
        """
        target_device = device or self.device
        target_compute = compute_type or (self.compute_type if target_device == self.device else ("int8" if target_device == "cpu" else self.compute_type))
        cfg = (self.model_name, target_device, target_compute, self.device_index)

        # Fast path: already loaded with this config
        if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
            return

        async with _lock:
            if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
                return
            log.warning(
                "[WhisperService] loading Whisper: model=%s device=%s compute=%s idx=%d",
                cfg[0], cfg[1], cfg[2], cfg[3]
            )
            WhisperService._model = WhisperModel(
                self.model_name,
                device=cfg[1],
                device_index=cfg[3],
                compute_type=cfg[2],
            )
            WhisperService._loaded_cfg = cfg

    async def _ensure_model(self):
        """
        Ensure a model is loaded on the preferred device; if that fails due to GPU issues, load on CPU.
        """
        try:
            await self._load_model(device=self.device, compute_type=self.compute_type)
        except RuntimeError as e:
            if _is_cuda_alloc_error(str(e)):
                log.error("[WhisperService] CUDA load failed (%s). Falling back to CPU quantized.", e)
                await self._load_model(device="cpu", compute_type="int8")
            else:
                raise

    def _temp_suffix_for_content_type(self, content_type: str) -> str:
        ct = (content_type or "").lower()
        if "webm" in ct:            return ".webm"
        if "ogg" in ct or "opus" in ct: return ".ogg"
        if "wav" in ct:             return ".wav"
        if "m4a" in ct or "mp4" in ct or "aac" in ct: return ".m4a"
        if "mpeg" in ct or "mp3" in ct: return ".mp3"
        return ".wav"

    async def transcribe(
        self,
        wav_bytes: bytes,
        language: str = "en",
        initial_prompt: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> str:
        """
        Transcribe audio bytes → text.
        Robust against GPU OOM: retries once on CPU with lighter settings.
        """
        await self._ensure_model()

        # Write to a temp file so ffmpeg/ctranslate2 can sniff format reliably.
        suffix = self._temp_suffix_for_content_type(content_type)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name

        # app/services/stt/whisper_service.py  (inside _do_transcribe)
        def _do_transcribe() -> str:
            segments, info = WhisperService._model.transcribe(
                tmp,
                language=language,
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=self.beam_size,           # keep 1
                best_of=self.best_of,               # keep 1
                # keep temperature conservative; a tuple disables temp-sweep in some versions
                temperature=(self.temperature,) if isinstance(self.temperature, (int, float)) else self.temperature,
                initial_prompt=(initial_prompt[:1800] if initial_prompt else None),
                word_timestamps=False,

                # 👇 additions that help with repetitions:
                condition_on_previous_text=False,   # don't “carry over” context between chunks
                prompt_reset_on_temperature=True,   # safer if temp ever > 0
                compression_ratio_threshold=2.6,    # skip overly repetitive gibberish
                log_prob_threshold=-1.0,            # drop very low-confidence decodes
                no_speech_threshold=0.6,            # drop blank noise segments

                # memory knob
                chunk_length=self.chunk_length,     # ~10–15s is good
            )
            text = " ".join(s.text.strip() for s in segments if getattr(s, "text", None)).strip()
            text = _dedupe_sentences(text)

            # final belt-and-suspenders: cap runaway outputs (rare, but safe)
            if len(text) > 2000:
                text = text[:2000].rsplit(" ", 1)[0] + "…"

            return text



        try:
            try:
                return _do_transcribe()
            except RuntimeError as e:
                if _is_cuda_alloc_error(str(e)) and WhisperService._loaded_cfg and WhisperService._loaded_cfg[1] == "cuda":
                    log.error("[WhisperService] CUDA transcribe failed (%s). Retrying on CPU.", e)
                    # Reload on CPU and retry once with even safer settings
                    await self._load_model(device="cpu", compute_type="int8")
                    # tighten further for CPU path if desired
                    old_chunk = self.chunk_length
                    try:
                        self.chunk_length = min(self.chunk_length, 15)
                        return _do_transcribe()
                    finally:
                        self.chunk_length = old_chunk

                else:
                    raise
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
