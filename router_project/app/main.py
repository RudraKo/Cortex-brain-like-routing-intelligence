from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import chat
from app.db.database import init_db

app = FastAPI(title="Intelligent LLM Router API", version="1.0.0")

# Initialize database
init_db()

# Allow CORS for potential frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Intelligent LLM Router is running."}
