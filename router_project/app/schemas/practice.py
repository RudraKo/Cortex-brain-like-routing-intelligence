from pydantic import BaseModel, Field


class PromptAnalysisRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    goal: str | None = None


class PromptScores(BaseModel):
    clarity: float
    context: float
    constraints: float
    evaluation: float
    structure: float


class PromptAnalysisResponse(BaseModel):
    scores: PromptScores
    overall_score: float
    detected_task_type: str
    detected_complexity: float
    strengths: list[str]
    improvements: list[str]
    rewritten_prompt: str
    checklist: list[str]


class PromptTestRequest(BaseModel):
    scenario: str = Field(..., min_length=1)
    prompts: list[str] = Field(..., min_length=1, max_length=6)


class PromptTestMetrics(BaseModel):
    relevance: float
    instruction_adherence: float
    structure: float
    conciseness: float


class RouterMeta(BaseModel):
    model: str
    task_type: str
    complexity_score: float
    latency_sec: float
    total_tokens: int
    from_cache: bool


class PromptTestRun(BaseModel):
    index: int
    prompt: str
    response: str
    score: float
    metrics: PromptTestMetrics
    router_meta: RouterMeta


class PromptTestResponse(BaseModel):
    scenario: str
    winner_index: int | None
    ranking: list[int]
    runs: list[PromptTestRun]


class LearnerProgressUpdateRequest(BaseModel):
    completed_lessons: list[str] = Field(default_factory=list)
    active_level: str | None = None


class LearnerProgressResponse(BaseModel):
    learner_id: str
    completed_lessons: list[str]
    active_level: str | None
    total_lessons: int
    completed_count: int
    completion_pct: float
