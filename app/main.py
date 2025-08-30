from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import configure_logging
from app.api.routers import health, stt, tts, sessions
from app.api.routers import sim, ipa
from app.lifespan import lifespan

configure_logging(settings.LOG_LEVEL)
app = FastAPI(title="English Trainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(stt.router, prefix=settings.API_PREFIX)
app.include_router(tts.router, prefix=settings.API_PREFIX)
app.include_router(sessions.router, prefix=settings.API_PREFIX)
app.include_router(sim.router,      prefix=settings.API_PREFIX)
app.include_router(ipa.router,      prefix=settings.API_PREFIX)