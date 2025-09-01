# app/api/routers/tts.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from app.schemas.tts import TTSRequest
from app.core.config import settings

# Choose engine (kept compatible with your existing pattern)
if settings.TTS_ENGINE.lower() == "bark":
    from app.services.tts.bark_service import BarkService as TTSImpl
    _svc = TTSImpl(
        voice_preset=settings.BARK_VOICE,
        text_temp=settings.BARK_TEXT_TEMP,
        waveform_temp=settings.BARK_WAVEFORM_TEMP,
        use_small_default=getattr(settings, "BARK_USE_SMALL_DEFAULT", False),
    )
else:
    # Fallback if you keep Piper around
    from app.services.tts.piper_service import PiperService as TTSImpl
    _svc = TTSImpl(settings.PIPER_VOICE)

router = APIRouter(prefix="/tts", tags=["tts"])

@router.post("")
async def synth(req: TTSRequest):
    """
    Non-streaming TTS. Returns a single WAV buffer (audio/wav).
    Body: { text, voice?, use_small?, seed?, text_temp?, waveform_temp? }
    """
    try:
        wav: bytes = await _svc.synthesize_bytes(
            text=req.text,
            voice=req.voice,
            use_small=req.use_small,
            seed=req.seed,
            text_temp=req.text_temp,
            waveform_temp=req.waveform_temp,
        )
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'},
        )
    except Exception as e:
        # Add your logger here if desired
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

@router.post("/warm")
async def warm(use_small: bool | None = None):
    """
    Preload Bark models into memory. Pass use_small=true for small models.
    """
    try:
        # Only BarkService exposes warm_models; Piper can no-op if selected
        if hasattr(_svc, "warm_models"):
            _svc.warm_models(use_small=bool(use_small))
        return JSONResponse({"ok": True, "use_small": bool(use_small)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")

@router.get("/health")
async def health():
    """
    Quick health info. Useful for readiness/liveness checks.
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        info = {
            "ok": True,
            "engine": settings.TTS_ENGINE.lower(),
            "device": device,
            "bark_use_small_default": getattr(settings, "BARK_USE_SMALL_DEFAULT", False),
        }
        # Optional: expose BarkService warm flags if present
        for attr in ("_is_warmed_full", "_is_warmed_small"):
            if hasattr(_svc, attr):
                info[attr] = bool(getattr(_svc, attr))
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health failed: {e}")
@router.get("")
async def synth_query(text: str, voice: str | None = None):
    wav = await _svc.synthesize_bytes(text=text, voice=voice)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )
