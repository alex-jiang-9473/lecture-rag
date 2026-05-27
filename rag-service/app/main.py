from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskResponse, QuestionRequest
from .services import RAGService
from .config import settings

service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ensure service readiness during application startup
    await service.ensure_ready()
    yield


app = FastAPI(
    title="Lecture RAG API",
    description="FastAPI + SQLAlchemy + PostgreSQL RAG backend for lecture Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# startup handled via lifespan


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Lecture RAG API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}




@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: QuestionRequest) -> AskResponse:
    result = await service.answer(payload.question, top_k=payload.top_k)
    return AskResponse(**result)
