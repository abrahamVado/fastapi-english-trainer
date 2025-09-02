# app/services/stt/whisper_service.py
from __future__ import annotations
from typing import Optional, Tuple
import os, tempfile, asyncio, logging, re
from difflib import SequenceMatcher

import numpy as np
import torch
from faster_whisper import WhisperModel

log = logging.getLogger("whisper")
_lock = asyncio.Lock()

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v

def _is_cuda_alloc_error(msg: str) -> bool:
    m = msg.lower()
    return ("cudnn_status" in m or "device_allocation_failed" in m or
            "out of memory" in m or ("cuda" in m and "failed" in m))

def _normalize_for_compare(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.?!]+$", "", s)
    return s

def _dedupe_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out, last_norm = [], ""
    for p in parts:
        p_norm = _normalize_for_compare(p)
        if p_norm and (p_norm == last_norm or
                       (last_norm and SequenceMatcher(None, p_norm, last_norm).ratio() >= 0.92)):
            continue
        if p.strip():
            out.append(p.strip())
            last_norm = p_norm
    s = " ".join(out)
    s = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Try to import the high-quality decoder
try:
    from app.services.stt.audio_decode import decode_to_float32_mono_16k  # type: ignore
    _HAS_DECODER = True
except Exception:
    _HAS_DECODER = False

class WhisperService:
    """
    Faster-Whisper wrapper with:
      - CUDA→CPU fallback
      - Clean PCM path (PyAV) when available
      - Anti-repeat decoding options
      - Env-configurable knobs
    ENV (optional):
      WHISPER_MODEL (default: "base.en")
      WHISPER_DEVICE ("cuda"/"cpu", default: auto)
      WHISPER_DEVICE_INDEX (default: "0")
      WHISPER_COMPUTE ("float16"/"int8_float16"/"int8", default: auto)
      WHISPER_BEAM_SIZE (default: "5")
      WHISPER_BEST_OF (default: "5")
      WHISPER_TEMPERATURE (default: "0")
      WHISPER_CHUNK_LENGTH (seconds, default: "15")
    """
    _model: Optional[WhisperModel] = None
    _loaded_cfg: Optional[Tuple[str, str, str, int]] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        device_index: Optional[int] = None,
    ):
        # bump default a notch for quality
        self.model_name   = model_name   or _env("WHISPER_MODEL", "base.en")
        self.device       = device       or _env("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = int(device_index if device_index is not None else _env("WHISPER_DEVICE_INDEX", "0"))
        self.compute_type = compute_type or _env("WHISPER_COMPUTE",
                                "int8_float16" if self.device == "cuda" else "int8")

        # Better decoding defaults (you can still override via env)
        self.beam_size     = int(_env("WHISPER_BEAM_SIZE", "5"))
        self.best_of       = int(_env("WHISPER_BEST_OF", "5"))
        self.temperature   = float(_env("WHISPER_TEMPERATURE", "0"))
        self.chunk_length  = int(_env("WHISPER_CHUNK_LENGTH", "15"))

        log.warning(
            "[WhisperService] target model='%s' device=%s compute=%s index=%d (beam=%d best_of=%d temp=%.2f chunk=%ds)",
            self.model_name, self.device, self.compute_type, self.device_index,
            self.beam_size, self.best_of, self.temperature, self.chunk_length
        )

    async def _load_model(self, device: Optional[str] = None, compute_type: Optional[str] = None):
        target_device = device or self.device
        target_compute = compute_type or (self.compute_type if target_device == self.device
                                          else ("int8" if target_device == "cpu" else self.compute_type))
        cfg = (self.model_name, target_device, target_compute, self.device_index)
        if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
            return
        async with _lock:
            if WhisperService._model is not None and WhisperService._loaded_cfg == cfg:
                return
            log.warning("[WhisperService] loading: model=%s device=%s compute=%s idx=%d",
                        *cfg)
            WhisperService._model = WhisperModel(
                self.model_name,
                device=cfg[1],
                device_index=cfg[3],
                compute_type=cfg[2],
            )
            WhisperService._loaded_cfg = cfg

    async def _ensure_model(self):
        try:
            await self._load_model(device=self.device, compute_type=self.compute_type)
        except RuntimeError as e:
            if _is_cuda_alloc_error(str(e)):
                log.error("[WhisperService] CUDA load failed (%s). Fallback to CPU int8.", e)
                await self._load_model(device="cpu", compute_type="int8")
            else:
                raise

    def _temp_suffix_for_content_type(self, content_type: str) -> str:
        ct = (content_type or "").lower()
        if "webm" in ct: return ".webm"
        if "ogg" in ct or "opus" in ct: return ".ogg"
        if "wav" in ct: return ".wav"
        if "m4a" in ct or "mp4" in ct or "aac" in ct: return ".m4a"
        if "mpeg" in ct or "mp3" in ct: return ".mp3"
        return ".wav"

    async def transcribe(
        self,
        wav_bytes: bytes,                    # raw upload bytes (webm/ogg/wav/...)
        language: str = "en",
        initial_prompt: Optional[str] = None,
        content_type: str = "audio/webm",
    ) -> str:
        """
        Transcribe audio bytes → text with repetition guards.
        """
        await self._ensure_model()

        def _decode_to_array() -> tuple[np.ndarray, int]:
            if _HAS_DECODER:
                return decode_to_float32_mono_16k(wav_bytes)
            # Fallback: write tempfile and let ffmpeg inside faster-whisper handle it.
            suffix = self._temp_suffix_for_content_type(content_type)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(wav_bytes)
                return np.array([]), f.name  # special sentinel: path string

        # Decode
        audio_or_path = None
        sr = 16000
        try:
            decoded, sr = _decode_to_array()
            audio_or_path = decoded if decoded.size else None
        except Exception as e:
            log.warning("[WhisperService] PyAV decode failed (%s). Using path fallback.", e)
            audio_or_path = None  # force path fallback

        # Build transcribe kwargs
        common_kwargs = dict(
            language=language,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 150},
            beam_size=max(1, self.beam_size),
            best_of=max(1, self.best_of),
            temperature=[float(self.temperature)],     # no temp sweep
            initial_prompt=(initial_prompt[:1800] if initial_prompt else None),
            word_timestamps=False,
            condition_on_previous_text=False,          # <- critical
            no_repeat_ngram_size=4,                    # <- blocks short repeats
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            chunk_length=self.chunk_length,
            prepend_punctuations="¿([{-",
            append_punctuations=".,!?)}]％%",
        )

        # Do the transcription
        try:
            if isinstance(audio_or_path, np.ndarray) and audio_or_path.size:
                segments, info = WhisperService._model.transcribe(
                    audio=audio_or_path, **common_kwargs
                )
            else:
                # path fallback (write a temp file)
                suffix = self._temp_suffix_for_content_type(content_type)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(wav_bytes)
                    path = f.name
                segments, info = WhisperService._model.transcribe(path, **common_kwargs)
                try: os.remove(path)
                except Exception: pass
        except RuntimeError as e:
            if _is_cuda_alloc_error(str(e)) and WhisperService._loaded_cfg and WhisperService._loaded_cfg[1] == "cuda":
                log.error("[WhisperService] CUDA transcribe failed (%s). Retrying on CPU.", e)
                await self._load_model(device="cpu", compute_type="int8")
                # Tighten a bit for CPU
                common_kwargs["beam_size"] = max(1, min(3, self.beam_size))
                common_kwargs["best_of"]   = max(1, min(3, self.best_of))
                segments, info = WhisperService._model.transcribe(
                    audio=audio_or_path if isinstance(audio_or_path, np.ndarray) and audio_or_path.size else path,
                    **common_kwargs
                )
            else:
                raise

        text = " ".join(s.text.strip() for s in segments if getattr(s, "text", None)).strip()
        text = _dedupe_sentences(text)

        # Light normalization: capitalize first letter if sentence looks lowercased
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Trim pathological outputs
        if len(text) > 2000:
            text = text[:2000].rsplit(" ", 1)[0] + "…"

        # Optional: quick quality log
        try:
            avg_logprob = float(np.mean([getattr(s, "avg_logprob", 0.0) for s in segments]))
            log.info("[WhisperService] len=%.1fs tokens=%d avg_logprob=%.3f",
                     getattr(info, "duration", 0.0), getattr(info, "num_tokens", 0), avg_logprob)
        except Exception:
            pass

        return text
