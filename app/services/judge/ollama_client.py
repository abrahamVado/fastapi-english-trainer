class PiperService:
    def __init__(self, voice: str):
        self.voice = voice
    async def synthesize(self, text: str, voice: str | None = None):
        # stub: yields fake audio
        yield b"RIFF....WAVE"
