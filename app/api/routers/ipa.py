# app/api/routers/ipa.py
from fastapi import APIRouter
from app.schemas.ipa import (
    PronounceRequest, PronounceResponse, TokenResult,
    PronScoreRequest, PronScoreResponse
)
from app.services.ipa.mapping import tokenize, en_to_ipa, map_to_latam, respell, score_pronunciation

router = APIRouter(prefix="/ipa", tags=["ipa"])

@router.post("/pronounce", response_model=PronounceResponse)
def pronounce(req: PronounceRequest):
    opts = req.options or {}
    out=[]
    for tok in tokenize(req.text):
        eng = en_to_ipa(tok.lower())
        lat = map_to_latam(
            eng,
            theta=getattr(opts, "theta_mode", "t"),
            mode=getattr(opts, "mode", "strict"),
            r=getattr(opts, "r_variant", "tap"),
            schwa=getattr(opts, "schwa", "e"),
        )
        out.append(TokenResult(
            token=tok,
            english_ipa=eng,
            latam_ipa=lat,
            respelling=respell(lat) if req.respell else None
        ))
    return PronounceResponse(text=req.text, tokens=out)

@router.post("/pron/score", response_model=PronScoreResponse)
def pron_score(req: PronScoreRequest):
    opts = req.options or {}
    res = score_pronunciation(
        req.expected_text,
        req.heard_text,
        theta=getattr(opts, "theta_mode", "t"),
        mode=getattr(opts, "mode", "strict"),
        r=getattr(opts, "r_variant", "tap"),
        schwa=getattr(opts, "schwa", "e"),
    )
    return PronScoreResponse(**res)
