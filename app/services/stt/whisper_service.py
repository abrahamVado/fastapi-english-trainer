# app/services/stt/whisper_service.py
from __future__ import annotations
from faster_whisper import WhisperModel
from typing import Optional, Iterable
import tempfile, os

class WhisperService:
    """
    Production-friendly wrapper around faster-whisper.
    - Model name comes from env/config.
    - Accepts raw audio bytes, writes a temp file, transcribes.
    - Tuned for English interview context (bias prompt hook).
    """
    _model: Optional[WhisperModel] = None

    def __init__(self, model_name: str = "base", device: Optional[str] = None, compute_type: str = "auto"):
        self.model_name = model_name
        self.device = device or ("cuda" if os.getenv("USE_CUDA","0") == "1" else "cpu")
        self.compute_type = compute_type

    def _ensure_model(self):
        if WhisperService._model is None:
            WhisperService._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

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
            try: os.remove(tmp)
            except: pass
