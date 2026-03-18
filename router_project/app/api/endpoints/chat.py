import os

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import LearnerProgress, RequestLog
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.practice import (
    LearnerProgressResponse,
    LearnerProgressUpdateRequest,
    PromptAnalysisRequest,
    PromptAnalysisResponse,
    PromptTestRequest,
    PromptTestResponse,
)
from app.services.llm_router import route_request
from app.services.prompt_practice import (
    analyze_prompt_quality,
    calculate_completion,
    get_full_course_curriculum,
    get_learning_modules,
    progress_from_json,
    progress_to_json,
    run_prompt_test,
)

router = APIRouter()
WRITE_API_KEY = (os.getenv("PROMPT_STUDIO_WRITE_API_KEY") or "").strip()


def _require_write_access(x_api_key: str | None = Header(default=None)):
    """Optional lightweight auth: enforced only when PROMPT_STUDIO_WRITE_API_KEY is configured."""
    if not WRITE_API_KEY:
        return
    if x_api_key != WRITE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_require_write_access),
):
    try:
        result = route_request(request.prompt)

        # Only log to DB if it's a fresh (non-cached) request
        if not result.get("from_cache"):
            log_entry = RequestLog(
                prompt=request.prompt,
                task_type=result["task_type"],
                complexity_score=result["complexity_score"],
                chosen_model=result["model"],
                latency_sec=result["latency_sec"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
                retried=result["retried"],
                from_cache=False,
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)

        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Returns per-model aggregate stats: total requests, avg latency, avg tokens.
    """
    retried_expr = func.sum(case((RequestLog.retried == True, 1), else_=0)).label(
        "total_retries"
    )

    rows = (
        db.query(
            RequestLog.chosen_model,
            RequestLog.task_type,
            func.count(RequestLog.id).label("total_requests"),
            func.round(func.avg(RequestLog.latency_sec), 3).label("avg_latency_sec"),
            func.round(func.avg(RequestLog.total_tokens), 1).label("avg_tokens"),
            func.sum(RequestLog.total_tokens).label("total_tokens_used"),
            retried_expr,
        )
        .group_by(RequestLog.chosen_model, RequestLog.task_type)
        .order_by(RequestLog.chosen_model)
        .all()
    )

    return [
        {
            "model": r.chosen_model,
            "task_type": r.task_type,
            "total_requests": r.total_requests,
            "avg_latency_sec": r.avg_latency_sec,
            "avg_tokens": r.avg_tokens,
            "total_tokens_used": r.total_tokens_used,
            "total_retries": r.total_retries or 0,
        }
        for r in rows
    ]


@router.get("/practice/modules")
async def practice_modules():
    return get_learning_modules()


@router.get("/practice/course")
async def practice_course():
    return get_full_course_curriculum()


@router.post("/practice/analyze", response_model=PromptAnalysisResponse)
async def practice_analyze(
    request: PromptAnalysisRequest,
    _: None = Depends(_require_write_access),
):
    try:
        analyzed = analyze_prompt_quality(request.prompt, request.goal)
        return PromptAnalysisResponse(**analyzed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/practice/test", response_model=PromptTestResponse)
async def practice_test(
    request: PromptTestRequest,
    _: None = Depends(_require_write_access),
):
    try:
        tested = run_prompt_test(request.scenario, request.prompts)
        return PromptTestResponse(**tested)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{learner_id}", response_model=LearnerProgressResponse)
async def get_progress(learner_id: str, db: Session = Depends(get_db)):
    progress = db.query(LearnerProgress).filter(LearnerProgress.learner_id == learner_id).first()
    completed_lessons = progress_from_json(progress.completed_lessons_json if progress else None)
    total_lessons, completed_count, completion_pct = calculate_completion(completed_lessons)
    active_level = progress.active_level if progress else None

    return LearnerProgressResponse(
        learner_id=learner_id,
        completed_lessons=completed_lessons,
        active_level=active_level,
        total_lessons=total_lessons,
        completed_count=completed_count,
        completion_pct=completion_pct,
    )


@router.put("/progress/{learner_id}", response_model=LearnerProgressResponse)
async def put_progress(
    learner_id: str,
    request: LearnerProgressUpdateRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_require_write_access),
):
    all_levels = {level["id"] for level in get_full_course_curriculum()["levels"]}
    active_level = request.active_level if request.active_level in all_levels else None
    completed_lessons = request.completed_lessons
    total_lessons, completed_count, completion_pct = calculate_completion(completed_lessons)

    row = db.query(LearnerProgress).filter(LearnerProgress.learner_id == learner_id).first()
    if row is None:
        row = LearnerProgress(
            learner_id=learner_id,
            completed_lessons_json=progress_to_json(completed_lessons),
            active_level=active_level,
            completion_pct=completion_pct,
        )
        db.add(row)
    else:
        row.completed_lessons_json = progress_to_json(completed_lessons)
        row.active_level = active_level
        row.completion_pct = completion_pct

    db.commit()
    db.refresh(row)

    return LearnerProgressResponse(
        learner_id=learner_id,
        completed_lessons=progress_from_json(row.completed_lessons_json),
        active_level=row.active_level,
        total_lessons=total_lessons,
        completed_count=completed_count,
        completion_pct=row.completion_pct,
    )
