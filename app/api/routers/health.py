from fastapi import APIRouter
from app.core.version import APP_NAME, APP_VERSION

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def health():
    return {"name": APP_NAME, "version": APP_VERSION, "ok": True}
