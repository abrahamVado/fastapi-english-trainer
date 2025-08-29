from pydantic import BaseModel

class STTRequest(BaseModel):
    language: str | None = None

class STTResponse(BaseModel):
    text: str
    duration_sec: float | None = None
