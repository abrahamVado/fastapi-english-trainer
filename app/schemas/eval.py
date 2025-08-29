from pydantic import BaseModel

class EvalRequest(BaseModel):
    reference: str
    hypothesis: str

class EvalResponse(BaseModel):
    pronunciation: float
    fluency: float
    overall: float
