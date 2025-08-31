# app/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # warm Bark if configured
        try:
            from app.core.config import settings
            if settings.TTS_ENGINE.lower() == "bark":
                from app.services.tts.bark_service import BarkService
                BarkService.warm_models()
        except Exception as e:
            # don't crash the app if warmup fails
            print(f"[lifespan] Bark warmup skipped: {e}")
        yield
    finally:
        pass
