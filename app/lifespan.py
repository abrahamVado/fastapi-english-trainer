# app/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.tts.bark_service import get_bark_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # warm Bark on CPU with small models (safe + fast)
        get_bark_service().warm_models(use_small=True)
    except Exception as e:
        print(f"[lifespan] Bark warmup skipped: {e}")
    yield
