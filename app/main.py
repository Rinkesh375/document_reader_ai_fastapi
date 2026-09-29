from fastapi import FastAPI
from app.query_router import router as query_router

app = FastAPI(
    title="Intelligent RAG API",
    description="""
    A production-ready Retrieval-Augmented Generation (RAG) API
    for document ingestion, semantic search, and AI-powered
    question answering using relevant knowledge from your documents.
    """,
    version="1.0.0",
)

app.include_router(query_router)