from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class Embedder:    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)[0]

if __name__ == "__main__":
    embedder = Embedder()
    text = "HELLO"
    vector = embedder.embed_query(text)