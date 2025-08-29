from fastapi import APIRouter
from app.schemas.eval import EvalRequest, EvalResponse
from app.services.eval.speech_eval import EvalService

router = APIRouter(prefix="/eval", tags=["eval"])
svc = EvalService()

@router.post("", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    return svc.score(req.reference, req.hypothesis)
