from fastapi import APIRouter
from app.schemas.session import SessionCreate, SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Fake in-memory sessions
_sessions = {}

@router.post("", response_model=SessionOut)
async def create_session(data: SessionCreate):
    sid = len(_sessions)+1
    sess = SessionOut(id=sid, user_id=data.user_id, status="active")
    _sessions[sid] = sess
    return sess

@router.get("", response_model=list[SessionOut])
async def list_sessions():
    return list(_sessions.values())

@router.post("/{sid}/end", response_model=SessionOut)
async def end_session(sid: int):
    sess = _sessions.get(sid)
    if not sess: return {"error": "not found"}
    sess.status = "ended"
    return sess
