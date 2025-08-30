from fastapi import APIRouter, UploadFile, File
from app.schemas.stt import STTRequest, STTResponse
from app.services.stt.whisper_service import WhisperService
from app.core.config import settings

router = APIRouter(prefix="/stt", tags=["stt"])
svc = WhisperService(settings.WHISPER_MODEL)

@router.post("", response_model=STTResponse)
async def transcribe(payload: STTRequest | None = None, audio: UploadFile = File(...)):
    data = await audio.read()
    # bias prompt by role keywords later if you want
    text = await svc.transcribe(data, language=(payload.language if payload and payload.language else "en"))
    return STTResponse(text=text, duration_sec=None)
