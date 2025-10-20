"""
Embedding model wrapper for converting text to vectors.
Uses sentence-transformers for high-quality embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class Embedder:
    """
    Wrapper around sentence-transformers for generating embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name
                       'all-MiniLM-L6-v2' is fast and good quality (384 dims)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """
        Convert text(s) to embedding vector(s).
        
        Args:
            texts: Single string or list of strings
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of shape (n, dimension) for n texts
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convenience method for embedding a single query.
        Returns a 1D array.
        """
        return self.embed(query)[0]


if __name__ == "__main__":
    # Test the embedder
    embedder = Embedder()
    
    # Test with single text
    text = "Machine learning is awesome"
    vector = embedder.embed_query(text)
    print(f"\nTest embedding:")
    print(f"Text: '{text}'")
    print(f"Vector shape: {vector.shape}")
    print(f"First 5 values: {vector[:5]}")