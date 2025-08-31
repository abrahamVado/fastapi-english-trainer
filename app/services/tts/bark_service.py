# app/services/tts/bark_service.py
from __future__ import annotations
import io, re, wave, struct, os
import numpy as np

from bark import SAMPLE_RATE, generate_audio, preload_models

class BarkService:
    """
    Bark TTS wrapper.
    - Generates 24 kHz audio from text using a voice preset (history_prompt).
    - Returns a single WAV as a byte stream (you can stream chunks if you want later).
    """

    _is_warmed = False

    def __init__(self,
                 device: str = "cpu",
                 voice_preset: str = "v2/en_speaker_6",
                 text_temp: float = 0.7,
                 waveform_temp: float = 0.7):
        # Bark uses torch under the hood and will pick CUDA if available.
        # We keep 'device' for clarity; Bark API doesn't take it directly.
        os.environ.setdefault("SUNO_USE_SMALL_MODELS", "false")  # set "true" for lower VRAM
        self.voice_preset = voice_preset
        self.text_temp = float(text_temp)
        self.waveform_temp = float(waveform_temp)

    @staticmethod
    def warm_models():
        if not BarkService._is_warmed:
            # downloads / loads coarse, fine, and codec models into cache
            preload_models()
            BarkService._is_warmed = True

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
        # Bark returns float32 in [-1, 1]. Convert to 16-bit PCM WAV.
        samples = np.clip(samples, -1.0, 1.0)
        int16 = (samples * 32767.0).astype(np.int16)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int16.tobytes())
        bio.seek(0)
        return bio.read()

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        # Simple sentence splitter to keep Bark snappy on long inputs.
        # You can replace with a smarter splitter later.
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        # split at . ? ! ; or long commas
        parts = re.split(r"(?<=[\.\?\!;])\s+|,\s{1,3}", text)
        # Merge very short fragments with neighbors
        chunks, buf = [], ""
        for p in parts:
            if len((buf + " " + p).strip()) < 80:
                buf = (buf + " " + p).strip()
            else:
                if buf: chunks.append(buf)
                buf = p
        if buf: chunks.append(buf)
        return chunks

    async def synthesize(self, text: str, voice: str | None = None):
        """
        Async generator yielding a single WAV (you can later chunk-yield per sentence).
        """
        if not BarkService._is_warmed:
            BarkService.warm_models()

        preset = (voice or self.voice_preset) or "v2/en_speaker_6"
        chunks = self._split_into_chunks(text)

        # Generate per chunk and concatenate
        audio_all = None
        for ch in chunks or [""]:
            audio = generate_audio(
                ch,
                history_prompt=preset,
                text_temp=self.text_temp,
                waveform_temp=self.waveform_temp,
            )
            audio_all = audio if audio_all is None else np.concatenate([audio_all, audio])

        wav = self._to_wav_bytes(audio_all if audio_all is not None else np.zeros(1, dtype=np.float32))
        # Yield once (keeps signature compatible with StreamingResponse)
        yield wav
