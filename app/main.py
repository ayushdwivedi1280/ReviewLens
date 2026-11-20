"""FastAPI application factory and REST endpoints for ReviewLens.

Serves the interactive dashboard, processes CSV uploads, and generates on-demand replies.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings, ConfigError, Settings
from app.llm import LLMClient, LLMUnavailable, LLMRequestRejected
from app.cache import Cache
from app.cleaning import CleaningError
from app.pipeline import analyze_file, RUN_STORE
from app.replies import generate_suggested_reply

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ReplyRequest(BaseModel):
    """Payload for on-demand merchant reply generation."""
    run_id: str = Field(min_length=1)
    review_id: str = Field(min_length=1)


def create_app(
    llm: Optional[LLMClient] = None,
    cache: Optional[Cache] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="ReviewLens",
        description="E-commerce customer review intelligence and dashboard.",
        version="1.0.0",
    )

    # Injected or lazily initialized dependencies
    app.state.llm = llm
    app.state.cache = cache

    def get_active_llm() -> LLMClient:
        if app.state.llm is not None:
            return app.state.llm
        settings = get_settings()
        client = LLMClient(settings)
        app.state.llm = client
        return client

    def get_active_cache() -> Cache:
        if app.state.cache is not None:
            return app.state.cache
        try:
            settings = get_settings()
            cpath = settings.cache_path
        except ConfigError:
            cpath = "cache.db"
        cache_instance = Cache(cpath)
        app.state.cache = cache_instance
        return cache_instance

    @app.get("/api/health")
    def health_check():
        """Health check probe."""
        return {"status": "ok", "app": "ReviewLens"}

    @app.post("/api/analyze")
    async def analyze_reviews(file: UploadFile = File(...)):
        """Upload and analyze product reviews CSV."""
        # Read file bytes with a 2MB maximum limit
        contents = await file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds the 2 MB limit.",
            )

        # Retrieve LLM and Cache, handling missing API keys gracefully
        try:
            active_llm = get_active_llm()
            active_cache = get_active_cache()
            batch_size = active_llm.settings.batch_size
        except ConfigError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )

        try:
            result = analyze_file(
                file_bytes=contents,
                llm=active_llm,
                cache=active_cache,
                batch_size=batch_size,
            )
            return JSONResponse(content=result)
        except CleaningError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        except LLMRequestRejected as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM API rejected request: {exc}",
            )
        except LLMUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM service temporarily unavailable: {exc}",
            )

    @app.post("/api/reply")
    def generate_reply_endpoint(payload: ReplyRequest):
        """Generate an on-demand suggested merchant response for a needs_reply review."""
        run = RUN_STORE.get(payload.run_id)
        if not run or payload.review_id not in run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review '{payload.review_id}' for run '{payload.run_id}' not found in active session.",
            )

        review_info = run[payload.review_id]

        try:
            active_llm = get_active_llm()
        except ConfigError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )

        reply_text = generate_suggested_reply(
            llm=active_llm,
            product=review_info["product"],
            review_text=review_info["text"],
            sentiment=review_info["sentiment"],
            summary=review_info["summary"],
        )
        return {"reply": reply_text}

    # Serve static dashboard
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def serve_dashboard():
            return FileResponse(STATIC_DIR / "index.html")

    return app


# Default app instance for uvicorn
app = create_app()
