"""
Improvement 3 — FastAPI Backend
Serves recommendations from live raw_skills.csv via HTTP.
The HTML frontend fetches from this instead of using hardcoded JS data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

from recommendation_engine import RecommendationPipeline

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tech Stack Recommender API",
    description="DecodeLabs Project 3 — AI Recommendation Logic",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline once at startup ─────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
pipeline = RecommendationPipeline(top_n=3)
pipeline.load_dataset(CSV_PATH)

# ── Request / Response models ─────────────────────────────────────────────
class SkillsRequest(BaseModel):
    skills: List[str]
    top_n: int = 3

class RecommendationItem(BaseModel):
    rank: int
    role: str
    score: float
    percentage: float

class RecommendationResponse(BaseModel):
    status: str
    input_received: List[str]
    input_after_cleaning: List[str]
    recommendations: List[RecommendationItem]

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Tech Stack Recommender API is live!",
        "docs": "/docs",
        "recommend_endpoint": "/recommend",
    }

@app.get("/health")
def health():
    return {"status": "ok", "job_roles_loaded": len(pipeline.job_roles)}

@app.get("/roles")
def list_roles():
    """Return all available job roles and their skills."""
    return {
        "total": len(pipeline.job_roles),
        "roles": {role: skills for role, skills in pipeline.job_roles.items()},
    }

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: SkillsRequest):
    """
    Main recommendation endpoint.

    Body:
        { "skills": ["Python", "Cloud Computing", "K8s"], "top_n": 3 }

    Returns:
        Ranked list of matching job roles with cosine similarity scores.
    """
    if not request.skills:
        raise HTTPException(status_code=400, detail="skills list cannot be empty")

    # Dynamically set top_n per request
    pipeline.top_n = request.top_n

    try:
        from recommendation_engine import sanitize_input, apply_synonyms
        cleaned = sanitize_input(request.skills)
        synonymed = apply_synonyms(cleaned)

        results = pipeline.recommend(request.skills)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    items = [
        RecommendationItem(
            rank=i + 1,
            role=role,
            score=round(score, 4),
            percentage=round(score * 100, 1),
        )
        for i, (role, score) in enumerate(results)
    ]

    return RecommendationResponse(
        status="success",
        input_received=request.skills,
        input_after_cleaning=synonymed,
        recommendations=items,
    )
