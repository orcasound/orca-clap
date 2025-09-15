"""
app/models/text_embedding.py
Text embedding model and similarity search implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import polars as pl


from typing import List, Tuple


class TextEmbeddingModel:
    def __init__(self, cfg: dict):
        """
        This is completed upon space startup.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # load the dataset containing the embeddings, supported by polars: i.e. "hf://datasets/roneneldan/TinyStories/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
        self.dataset = pl.read_parquet(cfg['dataset_path']) # fields are audioUri, embedding

        # load the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        self.model = AutoModel.from_pretrained(cfg['model_name'])
        self.model.eval()
        self.model.to(self.device)


        # build the faiss index
        embeddings = np.vstack(self.dataset.select("embedding").to_numpy().squeeze())
        embedding_dimension = embeddings.shape[1]

        print("Embeddings shape:", embeddings.shape)

        embeddings = torch.tensor(embeddings).float()
        # embeddings = F.normalize(embeddings, p=2, dim=1).numpy()

        # calculate the norm v=v.max(∥v∥p​,ϵ)
        # norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        # eps = 1e-10
        # self.denom = embeddings.norm(2, 1, keepdim=True).clamp_min(eps)
        # self.normalized_embeddings = embeddings / self.denom.expand_as(embeddings)

        self.normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        print("Normalized embeddings shape:", self.normalized_embeddings.shape)

        print(embedding_dimension, self.normalized_embeddings.shape)

        assert len(embeddings.shape) == 2, "Embeddings should be 2D array"
        assert self.normalized_embeddings.shape[1] == embedding_dimension, "Embeddings shape mismatch"



    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.
        """
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embedded_text = self.model(**inputs)
        return embedded_text.pooler_output



    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the most similar audio embeddings to the text query.
        """

        if query.strip() == "":
            return []

        # Generate embedding for the query
        query_tensor = self.embed_text(query)
        query_tensor = query_tensor.reshape(1, -1).float()

        # Perform the search
        assert len(query_tensor.shape) == 2
        query_tensor = F.normalize(query_tensor, p=2, dim=1)

        distances = F.cosine_similarity(self.normalized_embeddings, query_tensor, dim=1).cpu().data.numpy()
        top_k_indices = np.argsort(-distances)[:top_k]

        # Retrieve the corresponding audio URIs and similarity scores
        audio_uris = self.dataset.select("audio_uri").to_numpy().flatten()
        
        results = [(audio_uris[idx], float(distances[idx])) for idx in top_k_indices]

        return results
    

# config data class
class TextEmbeddingConfig:
    def __init__(self, model_name:str, dataset_path:str):
        self.model_name = model_name
        self.dataset_path = dataset_path