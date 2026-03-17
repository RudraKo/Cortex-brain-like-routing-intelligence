from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    model: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    complexity_score: float
    task_type: str
    retried: bool
    from_cache: bool
