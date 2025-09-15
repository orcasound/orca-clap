"""
test_Server.py
convert a flac file to string
"""

import base64
import torchaudio
from torchcodec.encoders import AudioEncoder
import os
import numpy as np
import torch

import requests
import time

from typing import Iterator


def compress_audio_torchcodec(waveform, sampling_rate, format_type="flac", bit_rate=None):
    """
    Compress audio using TorchCodec AudioEncoder - no temporary files!
    
    Args:
        waveform: torch.Tensor of shape (1, num_samples) or (num_samples,)
        sampling_rate: int, sample rate
        format_type: str, format like "flac", "mp3", "wav", "ogg"
        bit_rate: int, optional bit rate for lossy formats
    
    Returns:
        bytes: compressed audio data
    """
    try:
        # Ensure correct tensor format for AudioEncoder
        if len(waveform.shape) == 2:
            # Already in (num_channels, num_samples) format
            if waveform.shape[0] == 1:
                samples = waveform  # Keep as is
            else:
                # Convert multi-channel to mono by averaging
                samples = waveform.mean(dim=0, keepdim=True)
        elif len(waveform.shape) == 1:
            # Add channel dimension: (num_samples,) -> (1, num_samples)
            samples = waveform.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")
        
        # Ensure values are in [-1, 1] range as required by AudioEncoder
        if samples.abs().max() > 1.0:
            print(f"Warning: Waveform values exceed [-1, 1] range. Max: {samples.abs().max()}")
            samples = samples.clamp(-1.0, 1.0)
        
        print(f"Encoding with shape: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Create AudioEncoder
        encoder = AudioEncoder(samples, sample_rate=sampling_rate)
        
        # Encode to tensor (returns uint8 tensor with raw bytes)
        if bit_rate is not None:
            encoded_tensor = encoder.to_tensor(format_type, bit_rate=bit_rate, num_channels=1, sample_rate=sampling_rate)
        else:
            encoded_tensor = encoder.to_tensor(format_type, num_channels=1, sample_rate=sampling_rate)
        
        # Convert uint8 tensor to bytes
        audio_bytes = encoded_tensor.numpy().tobytes()
        
        print(f"TorchCodec {format_type.upper()} encoding successful: {len(audio_bytes)} bytes")
        
        return audio_bytes
        
    except Exception as e:
        raise Exception(f"TorchCodec {format_type} encoding failed: {str(e)}")


def test_on_api(filename: str, chunk_size: int = 512 * 1024,):
    """
    post to aupi endpoint {audio: sampling_rate: token:}
    """

    print("API")

    # audio, sampling_rate, num_channels, num_frames = main(filename)

    waveform, sampling_rate = torchaudio.load(filename)
    # waveform = waveform[:, :int(sampling_rate*20)] 
    waveform = torchaudio.functional.resample(waveform, orig_freq=sampling_rate, new_freq=32000)
    sampling_rate = 32000  # Set the sample rate to 32000 Hz


    # Compress the audio
    compressed_audio = compress_audio_torchcodec(waveform, sampling_rate, format_type="flac")

    # STREAMING
    def generate_chunks() -> Iterator[bytes]:
        """Generate audio chunks for streaming"""
        for i in range(0, len(compressed_audio), chunk_size):
            chunk = compressed_audio[i:i + chunk_size]
            yield chunk
    
    headers = {
        'Content-Type': 'application/octet-stream',
        'X-Sampling-Rate': str(sampling_rate),
        'X-Compression': 'flac',
        'X-Audio-Length': str(len(compressed_audio))
    }
    
    print(f"Streaming {len(compressed_audio)} bytes in ~{len(compressed_audio)//chunk_size + 1} chunks")
    
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/embed_audio",
        data=generate_chunks(),
        headers=headers,
        stream=True  # Important for large responses
    )
    
    upload_time = time.time() - start_time
    print(f"Upload completed in {upload_time:.2f} seconds")


    print(response)
    print(response.status_code)
    print(response.json())





if __name__ == "__main__":
    import sys

    filename = sys.argv[1]
    test_on_api(filename)
    
