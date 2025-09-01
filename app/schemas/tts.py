# app/schemas/tts.py
from pydantic import BaseModel, Field, constr

class TTSRequest(BaseModel):
    text: constr(min_length=1, max_length=2000)
    voice: str | None = None
    use_small: bool | None = None
    seed: int | None = None
    text_temp: float | None = Field(None, ge=0.1, le=1.5)
    waveform_temp: float | None = Field(None, ge=0.1, le=1.5)
