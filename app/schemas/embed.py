from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class EmbeddingResponse(BaseModel):
    """
    Response body for text embedding.
    The return object is a list of tuples of audio files and their similarity scores.

    """
    results: List[Tuple[str, float]]
    energy_consumed_kwh: float
    co2_emissions_grams: float


class TextRequest(BaseModel):
    """
    Request body for text embedding.
    """
    text: str
    topk: int = 5


class StreamingAudioRequest(BaseModel):
    """Model for streaming audio request headers"""
    sampling_rate: int = Field(..., description="Audio sampling rate in Hz")
    compression: str = Field(default="flac", description="Audio compression format")
    audio_length: Optional[int] = Field(None, description="Total audio length in bytes")
    top_k: int = Field(default=5, description="Number of top similar audio files to return")
    
    # @field_validator('sampling_rate')
    def validate_sampling_rate(self, v):
        """
        Validate sampling rate is within acceptable range.
        """
        if v < 32000:
            raise ValueError('Sampling rate is too low for the model')
        if v > 192000:  # Reasonable upper limit
            raise ValueError('Sampling rate too high')
        return v
    
    # @field_validator('compression')
    def validate_compression(self, v):
        """
        Validate compression format.
        """
        allowed_formats = ['raw', 'flac', 'wav', 'mp3', 'ogg']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Compression must be one of: {allowed_formats}')
        return v.lower()
