# app/services/stt/whisper_service.py
from __future__ import annotations
from faster_whisper import WhisperModel
from typing import Optional
import tempfile, os
import torch


class WhisperService:
    """
    Production-friendly wrapper around faster-whisper.
    - Auto-selects GPU if available (torch.cuda).
    - Can be overridden with env USE_CUDA=0/1.
    - Keeps a singleton model for efficiency.
    """
    _model: Optional[WhisperModel] = None

    def __init__(self, model_name: str = "base", device: Optional[str] = None, compute_type: str = "auto"):
        self.model_name = model_name

        # Env override first
        use_cuda_env = os.getenv("USE_CUDA")

        if device:
            self.device = device
        elif use_cuda_env == "1":
            self.device = "cuda"
        elif use_cuda_env == "0":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # choose compute type
        if compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8_float16"
        else:
            self.compute_type = compute_type

        print(f"[WhisperService] loading model '{self.model_name}' on device={self.device}, compute_type={self.compute_type}")

    def _ensure_model(self):
        if WhisperService._model is None:
            WhisperService._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )

    async def transcribe(self, wav_bytes: bytes, language: str = "en", initial_prompt: Optional[str] = None) -> str:
        self._ensure_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            segments, info = WhisperService._model.transcribe(
                tmp,
                language=language,
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0,
                initial_prompt=initial_prompt[:1800] if initial_prompt else None,
                word_timestamps=False
            )
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
