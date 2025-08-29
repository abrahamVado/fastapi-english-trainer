from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.tts import TTSRequest
from app.services.tts.piper_service import PiperService

router = APIRouter(prefix="/tts", tags=["tts"])
svc = PiperService("en_US-amy-medium.onnx")

@router.post("", response_class=StreamingResponse)
async def synth(req: TTSRequest):
    async def _gen():
        async for chunk in svc.synthesize(req.text, req.voice):
            yield chunk
    return StreamingResponse(_gen(), media_type="audio/wav")
