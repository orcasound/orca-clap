"""
main.py

A FastAPI application for embedding text and audio submissions and returning similar content

Combined endpoints:
- /embed_text (from text_embedding.py)
- /embed_audio (from audio_embedding.py)
"""

# backend deps
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

# Import endpoint routers
from endpoints.embed_text import router as text_router
from endpoints.embed_audio import router as audio_router
import torch

# Import models for initialization
from core.config import settings  # Assuming you have a settings object
from models.text_embedding import TextEmbeddingModel, TextEmbeddingConfig
from models.audio_embedding import AudioEmbeddingModel, AudioEmbeddingConfig


# organization deps
from codecarbon import EmissionsTracker, core

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Combined application lifespan manager."""
    
    logger.info("Starting combined text and audio retrieval service")
    
    # Initialize text embedding model
    text_config = TextEmbeddingConfig(
        model_name=settings.text_model_name,
        dataset_path=settings.text_dataset_path
    )
    app.state.text_embedding_model = TextEmbeddingModel(cfg=text_config.__dict__)
    
    # Initialize audio embedding model
    audio_config = AudioEmbeddingConfig(
        model_name=settings.audio_model_name,
        dataset_path=settings.audio_dataset_path
    )
    app.state.audio_embedding_model = AudioEmbeddingModel(cfg=audio_config.__dict__)
    
    # Initialize emissions tracker
    tracker = EmissionsTracker(save_to_file=False, save_to_api=False)
    if not torch.cuda.is_available():
        core.cpu.is_psutil_available = lambda: False
    app.state.tracker = tracker
    
    yield
    
    # Cleanup
    del app.state.text_embedding_model
    del app.state.audio_embedding_model
    del app.state.tracker
    
    logger.info("Shutting down combined retrieval service")

app = FastAPI(
    lifespan=lifespan,
    title=settings.api_title,
    description="A service to query using natural language and audio over the orcasound database",
    version=settings.api_version,
)

app.include_router(text_router)
app.include_router(audio_router)
