import chromadb


CHROMA_DB_DIR = "chroma_db"

class VectorDB:
    def __init__(self, persist_directory: str = CHROMA_DB_DIR):
        """
        Initialize ChromaDB (Persistent client).
        Creates or loads a collection called 'rag_docs'.
        """
        # ✅ Create persistent client (no more Settings)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # ✅ Try to get collection, else create one
        try:
            self.collection = self.client.get_collection("rag_docs")
            print("ℹ️ Loaded existing ChromaDB collection: rag_docs")
        except Exception:
            self.collection = self.client.create_collection("rag_docs")
            print("🆕 Created new ChromaDB collection: rag_docs")

    def add_embeddings(self, docs_embeddings: list):
        valid_docs = [d for d in docs_embeddings if d.get("embedding") is not None]

        if not valid_docs:
            print("⚠️ No valid embeddings to add.")
            return

        ids = [str(d["id"]) for d in valid_docs]
        embeddings = [d["embedding"].tolist() for d in valid_docs]

        # Metadata: replace None or empty with safe defaults
        metadatas = [
            {
                "page": int(d["page"]) if d.get("page") is not None else 0,
                "chunk_id": int(d["chunk_id"]) if d.get("chunk_id") is not None else 0,
                "image_paths": str(d["image_paths"]) if d.get("image_paths") is not None else ""
            }
            for d in valid_docs
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"✅ Added {len(ids)} embeddings to ChromaDB")


    def query(self, query_emb, top_k: int = 5):
        """
        Query ChromaDB with a given embedding.
        Returns top_k most similar results.
        """
        if query_emb is None:
            raise ValueError("❌ query_emb cannot be None")

        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
        )
        return results