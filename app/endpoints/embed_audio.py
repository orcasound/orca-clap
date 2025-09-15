"""
app.py

A FastAPI application for embedding text submissions and returning similar audio

access points:
non_auth: /embed_text

"""

# backend deps
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import ValidationError
from schemas.embed import StreamingAudioRequest, EmbeddingResponse
from typing import List, Tuple, Optional, Union

# computation deps
from models.audio_embedding import AudioEmbeddingModel, AudioEmbeddingConfig
import torch
from torchcodec.decoders import AudioDecoder
from codecarbon import EmissionsTracker, core
import polars as pl
import base64

# organization deps
import os
from contextlib import asynccontextmanager
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)

def decompress_audio_torchcodec(audio_bytes: bytes, compression: str, target_sampling_rate: int):
    """
    Decompress audio using TorchCodec (no temporary files needed!)
    Returns: 
        audio tensor for compatibility with existing handler
    """
    if compression.lower() == "raw":
        # Audio is already raw PCM data
        decoded_audio = base64.b64decode(audio_bytes)
        raise NotImplementedError("Check the base64 decoding is working first")
        return decoded_audio
    
    elif compression.lower() in ["flac", "wav", "mp3", "ogg"]:
        try:
            # Use TorchCodec to decode directly from bytes - no temp files!
            decoder = AudioDecoder(
                source=audio_bytes,
                sample_rate=target_sampling_rate,  # Automatically resample to target rate
                num_channels=1  # Force mono output
            )
            
            # Decode all audio at once
            # The decoder automatically handles resampling and channel conversion
            waveform=decoder.get_all_samples()
            waveform = waveform.data
            
            # Ensure correct shape for your handler (1, samples)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            elif waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # Average channels if somehow multi-channel

            return waveform
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"TorchCodec failed to decompress {compression} audio: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported compression format: {compression}")

router = APIRouter()


@router.post("/embed_audio", response_model=EmbeddingResponse)
async def embed_audio(request: Request):
    """
    Embed the input text and return the top-k most similar audio files.

    Args:
        request (TextRequest): The request body containing the text to embed and the number of top results to return.
    Returns:
        EmbeddingResponse: The response body containing the list of tuples of audio files and their similarity
        scores.
    """

    tracker = request.app.state.tracker
    embedding_model = request.app.state.audio_embedding_model

    try:
        # Extract and validate headers using Pydantic
        header_data = {
            'sampling_rate': request.headers.get('X-Sampling-Rate'),
            'compression': request.headers.get('X-Compression', 'raw'),
            'audio_length': request.headers.get('X-Audio-Length'),
            'top_k': request.headers.get('X-Top-K', 5)
        }
        # Remove None values for optional fields
        header_data = {k: v for k, v in header_data.items() if v is not None}

        # Validate with Pydantic
        try:
            validated_params = StreamingAudioRequest(**header_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Header validation error: {e}")
        
        # Use validated parameters
        sampling_rate = validated_params.sampling_rate
        compression = validated_params.compression
        audio_length = validated_params.audio_length
        top_k = validated_params.top_k

        print(f"Validated params - compression: {compression}, sampling_rate: {sampling_rate}, expected_length: {audio_length}")
        
        
        # Validate content type
        content_type = request.headers.get('content-type', '')
        if not content_type.startswith('application/octet-stream'):
            raise HTTPException(
                status_code=400, 
                detail=f"Content-Type must be 'application/octet-stream', got: {content_type}"
            )
        
        tracker.start_task()
        
        # Stream audio data into buffer
        audio_buffer = bytearray()
        chunk_count = 0
        total_bytes = 0

        try:
            async for chunk in request.stream():
                if chunk:  # Ignore empty chunks
                    audio_buffer.extend(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    
                    # Optional: Log progress for very large files
                    if chunk_count % 500 == 499:  # Log every 500 chunks
                        # logger.debug(f"Received chunk {chunk_count}, total bytes: {total_bytes}")
                        logger.debug("Received chunk %d, total bytes: %d", chunk_count, total_bytes)
                        
        except Exception as e:
            tracker.stop_task()
            raise HTTPException(status_code=400, detail=f"Error reading audio stream: {str(e)}")
        
        if len(audio_buffer) == 0:
            tracker.stop_task()
            raise HTTPException(status_code=400, detail="No audio data received in stream")
        
        # logger.debug(f"Stream complete - received {chunk_count} chunks, {total_bytes} total bytes")
        logger.debug("Stream complete - received %d chunks, %d total bytes", chunk_count, total_bytes)
        
        # Convert buffer to bytes for processing
        audio_bytes = bytes(audio_buffer)
        
        # Decompress audio using existing function
        waveform =  decompress_audio_torchcodec(audio_bytes, compression, sampling_rate)

    except HTTPException:
        if 'tracker' in locals():
            tracker.stop_task()
        raise
    except Exception as e:
        if 'tracker' in locals():
            tracker.stop_task()
        raise HTTPException(status_code=400, detail=f"Error processing audio stream: {str(e)}")
    
    # we now have the audio object in waveform
    try:
        results = embedding_model.search(
            waveform,
            sampling_rate=sampling_rate,
            top_k = top_k
        )

        task_emissions_data = tracker.stop_task()
    except Exception as e:
        tracker.stop_task()
        raise HTTPException(status_code=500, detail=f"Error extracting embeddings: {str(e)}")
    

    energy_consumed_kwh = task_emissions_data.energy_consumed
    co2_emissions_grams = task_emissions_data.emissions * 1000
    
    return EmbeddingResponse(results=results, energy_consumed_kwh=energy_consumed_kwh, co2_emissions_grams=co2_emissions_grams)
