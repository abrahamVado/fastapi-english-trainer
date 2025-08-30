from pydantic import BaseModel, Field
from typing import Optional, List

class SimStartReq(BaseModel):
    role: str = Field(..., examples=["node-react"])
    level: str = Field(..., examples=["junior","mid","senior"])
    mode: str = Field(..., examples=["technical","behavioral","mixed"])

class SimStartRes(BaseModel):
    session_id: str
    question_id: str
    question: str

class SimNextReq(BaseModel):
    session_id: str

class SimNextRes(BaseModel):
    question_id: str
    question: str

class SimAnswerTextReq(BaseModel):
    session_id: str
    question_id: str
    text: str

class SimAnswerAudioRes(BaseModel):
    session_id: str
    question_id: str
    asr_text: str
    confidence: float | None = None

class SimScoreReq(BaseModel):
    session_id: str
    question_id: str
    expected_text: str | None = None  # NEW: if provided, we do strict pronunciation scoring

class ScoreBreakdown(BaseModel):
    content: int
    pronunciation: int
    fluency: int
    overall: int

class SimScoreRes(BaseModel):
    scores: ScoreBreakdown
    tips: List[str] = []

class SimReportRes(BaseModel):
    session_id: str
    turns: list[dict]
    overall_avg: int
