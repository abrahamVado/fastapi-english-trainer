# app/api/routers/tts.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.tts import TTSRequest
from app.core.config import settings

# Choose engine
if settings.TTS_ENGINE.lower() == "bark":
    from app.services.tts.bark_service import BarkService as TTSImpl
    _svc = TTSImpl(
        voice_preset=settings.BARK_VOICE,
        text_temp=settings.BARK_TEXT_TEMP,
        waveform_temp=settings.BARK_WAVEFORM_TEMP,
    )
else:
    # fallback to your old Piper stub to avoid breaking dev
    from app.services.tts.piper_service import PiperService as TTSImpl
    _svc = TTSImpl(settings.PIPER_VOICE)

router = APIRouter(prefix="/tts", tags=["tts"])

@router.post("", response_class=StreamingResponse)
async def synth(req: TTSRequest):
    async def _gen():
        async for chunk in _svc.synthesize(req.text, req.voice):
            yield chunk
    # Bark returns 24kHz mono WAV
    return StreamingResponse(_gen(), media_type="audio/wav")
