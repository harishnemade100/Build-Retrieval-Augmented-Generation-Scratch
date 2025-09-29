from typing import Any
import numpy as np
from src.store_vector_database import VectorDB
from transformers import AutoModel, AutoTokenizer

class Retriever:
    """
    Retrieve relevant document chunks (text or images) from ChromaDB using embeddings.
    """

    def __init__(self, vector_db: VectorDB, embedding_model_name="jinaai/jina-clip-v1"):
        self.vector_db = vector_db
        self.model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert query text into an embedding using Jina-CLIP-v1.
        """
        return self.model.encode_text([query])[0]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve top-k document chunks relevant to the query.
        Returns metadata and embeddings.
        """
        query_emb = self.embed_query(query)
        results = self.vector_db.query(query_emb=query_emb, top_k=top_k)

        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                "id": results['ids'][0][i],
                "embedding": results['embeddings'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return retrieved_docs
