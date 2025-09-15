"""
app.py

A FastAPI application for embedding text submissions and returning similar audio

access points:
non_auth: /embed_text

"""

# backend deps
from fastapi import APIRouter, HTTPException, Request
from schemas.embed import TextRequest, EmbeddingResponse
# organization deps
import logging


logger = logging.getLogger(__name__)


# Create router
router = APIRouter()



@router.post("/embed_text", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest, app_request: Request):
    """
    Embed the input text and return the top-k most similar audio files.

    Args:
        request (TextRequest): The request body containing the text to embed and the number of top results to return.
    Returns:
        EmbeddingResponse: The response body containing the list of tuples of audio files and their similarity
        scores.
    """

    text = request.text
    topk = request.topk

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    if topk <= 0:
        raise HTTPException(status_code=400, detail="topk must be a positive integer.")

    embedding_model = app_request.app.state.text_embedding_model
    tracker = app_request.app.state.tracker

    # Start tracking emissions
    tracker.start_task()

    try:
        results = embedding_model.search(query=text, top_k=topk)
    finally:
        # Stop tracking emissions
        emissions = tracker.stop_task()
    logger.info(f"Emissions for embedding request: {emissions} kg CO2")

    energy_consumed_kwh = emissions.energy_consumed
    co2_emissions_grams = emissions.emissions * 1000

    return EmbeddingResponse(
        results=results,
        energy_consumed_kwh=energy_consumed_kwh,
        co2_emissions_grams=co2_emissions_grams
    )