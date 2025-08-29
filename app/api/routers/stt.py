from fastapi import APIRouter, UploadFile, File
from app.schemas.stt import STTRequest, STTResponse
from app.services.stt.whisper_service import WhisperService

router = APIRouter(prefix="/stt", tags=["stt"])

svc = WhisperService("base")

@router.post("", response_model=STTResponse)
async def transcribe(payload: STTRequest | None = None, audio: UploadFile = File(...)):
    data = await audio.read()
    text = await svc.transcribe(data)
    return STTResponse(text=text, duration_sec=None)
