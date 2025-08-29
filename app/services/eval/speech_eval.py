from app.schemas.eval import EvalResponse

class EvalService:
    def score(self, ref: str, hyp: str) -> EvalResponse:
        pron, flu = 0.8, 0.7
        return EvalResponse(pronunciation=pron, fluency=flu, overall=(pron+flu)/2)
