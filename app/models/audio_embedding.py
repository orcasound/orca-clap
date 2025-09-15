"""
app/models/audio_embedding.py
Audio embedding model and similarity search implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import numpy as np
import polars as pl
import torchaudio

from typing import List, Tuple, Union
from copy import deepcopy
import logging


logger = logging.getLogger(__name__)

class WhisperEmbeddingExtractor:
    """Main class for extracting embeddings from Whisper models."""
    
    def __init__(self, model_dir: str, first_entries: int = 1000, pooling_size: int = 10):
        self.model_dir = model_dir
        self.first_entries = first_entries
        self.pooling_size = pooling_size

        model = AutoModel.from_pretrained(model_dir)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_dir)

        logger.info(f"Successfully loaded model from %s", model_dir)

        # Move to appropriate device (skip for quantized models)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        self.model = model
        self.preprocessor = preprocessor

        
    
    def transform_audio(
        self, 
        waveform: torch.Tensor, 
        preprocessor: AutoFeatureExtractor, 
        input_sr: int = 16000
    ) -> torch.Tensor:
        """Transform audio waveform to model input features."""
        # Handle stereo to mono conversion
        if len(waveform.shape) == 2:
            waveform = waveform[0]
        
        # Convert to numpy for preprocessor
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # if the length of waveform_np is more than input_sr * 30 seconds, then we need to split the data into batches
        if len(waveform_np) > input_sr * 30:
            logger.info("Input audio is longer than 30 seconds, splitting into batches")
            waveform_np = [waveform_np[i:i + input_sr * 30] for i in range(0, len(waveform_np), input_sr * 30)]
        
        input_features = preprocessor(
            waveform_np,
            return_tensors="pt",
            sampling_rate=input_sr
        ).input_features.squeeze(0)

        # if the clip was short, we want to add a batch dimension
        if len(input_features.shape) == 2:
            input_features = input_features.unsqueeze(0)

        return input_features
    
    def extract_embeddings(
        self,
        batch_features: torch.Tensor
    ) -> torch.Tensor:
        """Extract embeddings from model encoder."""
        dtype = next(self.model.parameters()).dtype
        
        # Move to device and ensure correct dtype
        batch_features = batch_features.to(device=self.device, dtype=dtype)


        
        with torch.no_grad():
            # Get encoder output
            encoder_output = self.model.encoder(batch_features)
            last_hidden = encoder_output.last_hidden_state

            
            
            # Truncate to first N entries (for consistent timing)
            last_hidden = last_hidden[:, :self.first_entries]

            
            # Adaptive average pooling over time dimension
            # Shape: (batch, seq_len, hidden_dim) -> (batch, pooling_size, hidden_dim)
            pooled = torch.nn.AdaptiveAvgPool1d(self.pooling_size)(
                last_hidden.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
            ).permute(0, 2, 1)  # (batch, pooling_size, hidden_dim)


            # average along the batch dimension, since these batches are from the same sample
            pooled = pooled.mean(dim=0)  # (pooling_size, hidden_dim)

            # average along the pooling dimension to get a single embedding vector
            pooled = pooled.mean(dim=0)  # (hidden_dim,)

        return pooled
    
    def process_audio(
        self, 
        audio_data: Union[List[float], np.ndarray, torch.Tensor],
        sampling_rate: int
    ) -> torch.Tensor:
        """Process audio and extract embeddings with timing information."""
        
        # Convert audio data to tensor
        if not isinstance(audio_data, torch.Tensor):
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        else:
            audio_tensor = deepcopy(audio_data)

        
        # Resample if needed
        if sampling_rate != 32000:
            audio_tensor = torchaudio.functional.resample(
                waveform=audio_tensor.unsqueeze(0),  # Add batch dim
                orig_freq=sampling_rate,
                new_freq=32000
            ).squeeze(0)

        
        # Transform audio to features
        input_features = self.transform_audio(
            audio_tensor, 
            self.preprocessor, 
            16000
        )

        # Add batch dimension
        if len(input_features.shape) == 2:
            batch_features = input_features.unsqueeze(0)
        else:
            batch_features = input_features
        
        # Extract embeddings
        embeddings = self.extract_embeddings(batch_features)
        return embeddings
    


# config data class
class AudioEmbeddingConfig:
    def __init__(self, model_name:str, dataset_path:str):
        self.model_name = model_name
        self.dataset_path = dataset_path


class AudioEmbeddingModel:
    def __init__(self, cfg: dict):
        """
        This is completed upon space startup.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # load the dataset containing the embeddings, supported by polars: i.e. "hf://datasets/roneneldan/TinyStories/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
        self.dataset = pl.read_parquet(cfg['dataset_path']) # fields are audioUri, embedding

        # load the embedding model
        self.embedding_model = WhisperEmbeddingExtractor(cfg['model_name'])

        # build the index
        embeddings = np.vstack(self.dataset.select("embeddings_list").to_numpy().squeeze())
        embedding_dimension = embeddings.shape[1]

        logger.debug("Embeddings shape: %s", embeddings.shape)

        embeddings = torch.tensor(embeddings).float()

        # calculate the norm v=v.max(∥v∥p​,ϵ)
        self.normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        logger.debug("Normalized embeddings shape: %s", self.normalized_embeddings.shape)
        logger.debug("Embedding dimension: %d, Normalized embeddings shape: %s", embedding_dimension, self.normalized_embeddings.shape)

        assert len(embeddings.shape) == 2, "Embeddings should be 2D array"
        assert self.normalized_embeddings.shape[1] == embedding_dimension, "Embeddings shape mismatch"






    def search(self, waveform: torch.Tensor, sampling_rate: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the most similar audio embeddings to the text query.
        """

        # Generate embedding for the query
        query_tensor = self.embedding_model.process_audio(
            waveform,
            sampling_rate=sampling_rate
        )

        query_tensor = query_tensor.reshape(1, -1).float()

        # Perform the search
        assert len(query_tensor.shape) == 2
        query_tensor = F.normalize(query_tensor, p=2, dim=1)

        distances = F.cosine_similarity(self.normalized_embeddings, query_tensor, dim=1).cpu().data.numpy()
        top_k_indices = np.argsort(-distances)[:top_k]

        # Retrieve the corresponding audio URIs and similarity scores
        audio_uris = self.dataset.select("audioUri").to_numpy().flatten()
        
        results = [(audio_uris[idx], float(distances[idx])) for idx in top_k_indices]

        return results
    