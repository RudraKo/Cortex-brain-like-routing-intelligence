from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_router import route_request
from app.db.database import get_db
from app.db.models import RequestLog

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest, db: Session = Depends(get_db)):
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
        import traceback
        trace_str = traceback.format_exc()
        # include traceback in the 500 error for debugging Vercel
        raise HTTPException(status_code=500, detail=f"{repr(e)} | Trace: {trace_str}")


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Returns per-model aggregate stats: total requests, avg latency, avg tokens.
    """
    retried_expr = func.sum(
        case((RequestLog.retried == True, 1), else_=0)
    ).label("total_retries")

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
