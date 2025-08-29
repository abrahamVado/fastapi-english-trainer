from pydantic import BaseModel

class SessionCreate(BaseModel):
    user_id: int

class SessionOut(BaseModel):
    id: int
    user_id: int
    status: str
