# app/schemas/ipa.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Options(BaseModel):
    theta_mode: str = Field("t", description='"t" or "s" for /θ/')
    mode: str = Field("strict", description='"strict" or "approx"')
    r_variant: str = Field("tap", description='"tap" or "trill"')
    schwa: str = Field("e", description='"e" or "a"')

class TokenResult(BaseModel):
    token: str
    english_ipa: str
    latam_ipa: str
    respelling: Optional[str] = None

class PronounceRequest(BaseModel):
    text: str
    options: Optional[Options] = None
    respell: bool = True

class PronounceResponse(BaseModel):
    text: str
    tokens: List[TokenResult]

class PronScoreRequest(BaseModel):
    expected_text: str
    heard_text: str
    options: Optional[Options] = None

class PronScoreResponse(BaseModel):
    overall: dict
    words: list
