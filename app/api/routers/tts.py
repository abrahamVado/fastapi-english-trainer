# app/api/routers/tts.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from app.schemas.tts import TTSRequest
from app.core.config import settings

# Choose engine
if settings.TTS_ENGINE.lower() == "bark":
    from app.services.tts.bark_service import BarkService as TTSImpl
    _svc = TTSImpl(
        voice_preset=settings.BARK_VOICE,
        text_temp=settings.BARK_TEXT_TEMP,
        waveform_temp=settings.BARK_WAVEFORM_TEMP,
        use_small_default=getattr(settings, "BARK_USE_SMALL_DEFAULT", False),
    )
else:
    # If you keep Piper around, it can act as a simple fallback TTS.
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
        # Log if you have a logger; return sanitized error to client
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@router.get("")
async def synth_query(text: str, voice: str | None = None):
    """
    Convenience GET form: /api/tts?text=Hello&voice=v2/en_speaker_6
    Useful for quick manual tests in the browser.
    """
    try:
        wav = await _svc.synthesize_bytes(text=text, voice=voice)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@router.post("/warm")
async def warm(use_small: bool | None = None):
    """
    Preload Bark models into memory. Pass use_small=true for "small" models.

    IMPORTANT:
    - Bark has multiple versions with different preload signatures.
      This endpoint tolerates those differences and never crashes the app.
    - If Piper is selected, warm is a no-op (returns ok=True).
    """
    # If you’re not using Bark, just no-op successfully.
    if settings.TTS_ENGINE.lower() != "bark":
        return JSONResponse({"ok": True, "detail": "warmup not required for this TTS engine"})

    # Some BarkService implementations expose warm_models as @classmethod.
    warm_fn = getattr(_svc, "warm_models", None)
    if warm_fn is None:
        # Not exposed → treat as no-op
        return JSONResponse({"ok": True, "detail": "warm_models not available"})

    # Try with the provided flag first; fall back to no-arg call on TypeError.
    try:
        if use_small is None:
            warm_fn()  # tolerant Bark builds accept no args
        else:
            warm_fn(use_small=bool(use_small))
        return JSONResponse({"ok": True, "use_small": bool(use_small) if use_small is not None else None})
    except TypeError:
        # Signature mismatch (e.g., older Bark that accepts no kwargs)
        try:
            warm_fn()
            return JSONResponse({"ok": True, "use_small": None, "detail": "warm called without kwargs"})
        except Exception as e2:
            return JSONResponse({"ok": False, "detail": f"{e2}"})
    except Exception as e:
        # Do not 500 here; let the frontend proceed even if warm failed.
        return JSONResponse({"ok": False, "detail": f"{e}"})


@router.get("/health")
async def health():
    """
    Quick health check for the TTS subsystem (engine + device + warm flags).
    """
    try:
        info = {
            "ok": True,
            "engine": settings.TTS_ENGINE.lower(),
        }
        try:
            import torch  # optional diagnostic
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            info["device"] = "unknown"

        # Expose BarkService warm flags if present
        for attr in ("_is_warmed_full", "_is_warmed_small"):
            if hasattr(_svc, attr):
                info[attr] = bool(getattr(_svc, attr))
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health failed: {e}")
