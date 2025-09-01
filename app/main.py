# app/main.py

"""
Main FastAPI application entrypoint for the English Trainer API.

This file is responsible for:
  - Creating the FastAPI app instance.
  - Configuring logging.
  - Enabling CORS (Cross-Origin Resource Sharing) so the frontend can call the API.
  - Registering all routers (modular endpoint groups).
  - Exposing a simple root endpoint and a /health endpoint.
  - Attaching lifecycle hooks (startup/shutdown).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports (project-specific)
from app.core.config import settings               # Global settings (env-based, see config.py)
from app.core.logging import configure_logging     # Custom logging setup
from app.api.routers import (                      # All API routers (organized by feature)
    health,
    stt,
    tts,
    sessions,
    sim,
    ipa,
    gpu,
)
from app.lifespan import lifespan                  # Startup/shutdown event handler

from dotenv import load_dotenv
load_dotenv()  # will read .env in project root

# ---------------------------------------------------------------------
# 1. Configure logging
# ---------------------------------------------------------------------
# This sets up log formatting, levels, etc.
# Controlled by settings.LOG_LEVEL (e.g. "INFO", "DEBUG").
configure_logging(settings.LOG_LEVEL)


# ---------------------------------------------------------------------
# 2. Create FastAPI app instance
# ---------------------------------------------------------------------
# - `title`: Appears in the Swagger UI docs.
# - `version`: Optional version string (helps clients know API version).
# - `lifespan`: Hook for startup/shutdown tasks (e.g. preload models).
#
# NOTE: You can also move the docs under your API prefix if desired:
#   openapi_url=f"{settings.API_PREFIX}/openapi.json",
#   docs_url=f"{settings.API_PREFIX}/docs",
#   redoc_url=f"{settings.API_PREFIX}/redoc"
# ---------------------------------------------------------------------
app = FastAPI(
    title="English Trainer API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# 3. Configure CORS
# ---------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) allows your frontend (running at
# e.g. http://localhost:5173) to call this backend.
#
# - `allow_origins`: List of allowed origins (frontend URLs).
#   DO NOT use "*" if you also need credentials (cookies/Authorization headers).
#   In dev, you can set it to ["http://localhost:5173","http://localhost:3000"].
#
# - `allow_credentials`: Set True only if you specify explicit origins.
#
# - `allow_methods`: Which HTTP methods are allowed (["*"] = all).
# - `allow_headers`: Which headers are allowed (["*"] = all).
# --------------------------------------------------------------------
origins = ["https://trainer.local:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,        # set True only if you use cookies/auth headers
    allow_methods=["GET","POST","OPTIONS","PUT","PATCH","DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],          # optional
)

# ---------------------------------------------------------------------
# 4. Register Routers (modular endpoints)
# ---------------------------------------------------------------------
# Each router groups endpoints by feature/domain.
# All routers (except health) are included under the global API prefix
# (usually "/api", configurable via settings.API_PREFIX).
#
# Example final paths:
#   - /health                   (health router, no prefix, good for LB checks)
#   - /api/stt/...              (speech-to-text endpoints)
#   - /api/tts/...              (text-to-speech endpoints)
#   - /api/sessions/...         (session management)
#   - /api/sim/...              (simulation endpoints: start, answer, score, report)
#   - /api/ipa/...              (IPA / pronunciation endpoints)
#   - /api/gpu/...              (GPU info/testing endpoints)
# ---------------------------------------------------------------------
app.include_router(health.router)                        # available at /health
app.include_router(stt.router,      prefix=settings.API_PREFIX)
app.include_router(tts.router,      prefix=settings.API_PREFIX)
app.include_router(sessions.router, prefix=settings.API_PREFIX)
app.include_router(sim.router,      prefix=settings.API_PREFIX)
app.include_router(ipa.router,      prefix=settings.API_PREFIX)
app.include_router(gpu.router,      prefix=settings.API_PREFIX)
# ---------------------------------------------------------------------
# 5. Optional root endpoint (nice for humans)
# ---------------------------------------------------------------------
# Provides a friendly response if someone visits the base URL.
# Not included in schema/docs.
# ---------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "ok": True,
        "name": "English Trainer API",
        "health": "/health",          # LB-friendly check
        "docs": "/docs",              # Swagger UI
        "redoc": "/redoc",            # ReDoc UI
        "openapi": "/openapi.json",   # Raw OpenAPI spec
        "api_prefix": settings.API_PREFIX,
    }
