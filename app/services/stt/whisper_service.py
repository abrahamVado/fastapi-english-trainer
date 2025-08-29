class WhisperService:
    def __init__(self, model_name: str):
        self.model_name = model_name
    async def transcribe(self, wav_bytes: bytes) -> str:
        # stub: replace with faster-whisper
        return f"<transcribed {len(wav_bytes)} bytes>"
