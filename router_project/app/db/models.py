import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def utc_now_naive() -> datetime.datetime:
    """Return a UTC timestamp without tzinfo for naive DateTime columns."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=utc_now_naive)
    prompt = Column(String, index=True)
    task_type = Column(String, index=True)
    complexity_score = Column(Float)
    chosen_model = Column(String, index=True)
    latency_sec = Column(Float)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    retried = Column(Boolean, default=False)
    from_cache = Column(Boolean, default=False)


class LearnerProgress(Base):
    __tablename__ = "learner_progress"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String, unique=True, index=True, nullable=False)
    completed_lessons_json = Column(Text, default="[]")
    active_level = Column(String, nullable=True)
    completion_pct = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)
