import logging
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import chat
from app.db.database import init_db

app = FastAPI(title="Intelligent LLM Router API", version="1.0.0")

# Initialize database
init_db()

# Structured app logger for request and error tracing
logger = logging.getLogger("prompt_studio")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Allow CORS for potential frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limiter (per IP, per minute) for API routes
_rate_window_sec = 60
_rate_limit = int(os.getenv("PROMPT_STUDIO_RATE_LIMIT_PER_MINUTE", "120"))
_request_tracker: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path

    if _rate_limit > 0 and path.startswith("/api/"):
        now = time.time()
        bucket = _request_tracker[client_ip]
        while bucket and (now - bucket[0]) > _rate_window_sec:
            bucket.popleft()
        if len(bucket) >= _rate_limit:
            logger.warning("rate_limited ip=%s path=%s", client_ip, path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry shortly."},
            )
        bucket.append(now)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled_error method=%s path=%s ip=%s", request.method, path, client_ip)
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%s ip=%s",
        request.method,
        path,
        response.status_code,
        duration_ms,
        client_ip,
    )
    return response

# API router
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/")
def read_root():
    return {"message": "Intelligent LLM Router is running."}
