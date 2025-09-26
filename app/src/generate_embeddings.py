import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel
from app.src.store_vector_database import VectorDB

class EmbeddingsGenerator:
    """
    Generate embeddings for text and images using the official `jina-clip-v1` model API.
    """

    def __init__(self, metadata_path="data/attention_is_all_you_need/docs_metadata.json"):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{metadata_path} not found! Run ingestion first.")

        self.metadata_path = metadata_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.docs_metadata = json.load(f)

        # Load the Jina CLIP v1 model using the documented API
        self.model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)

    def generate(self):
        """
        Generate embeddings for all text and image chunks.
        Returns a list of dicts with embeddings.
        """
        docs_embeddings = []

        for doc in tqdm(self.docs_metadata, desc="Generating embeddings"):
            text = doc.get("text", "")
            image_paths = doc.get("image_paths", [])

            text_emb = None
            img_emb = None

            # Encode text if exists
            if text:
                # The model (AutoModel) offers `.encode_text(...)` per README usage :contentReference[oaicite:2]{index=2}
                text_emb = self.model.encode_text([text])[0]

            # Encode images if exists
            if image_paths:
                image_embs = []
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        # PIL image or path acceptable
                        image = Image.open(img_path).convert("RGB")
                        # The model offers `.encode_image(...)` per README API :contentReference[oaicite:3]{index=3}
                        emb = self.model.encode_image([image])[0]
                        image_embs.append(emb)
                if image_embs:
                    img_emb = np.mean(image_embs, axis=0)

            # Combine embeddings if both present (e.g. average)
            if text_emb is not None and img_emb is not None:
                emb = (text_emb + img_emb) / 2
            elif text_emb is not None:
                emb = text_emb
            else:
                emb = img_emb

            docs_embeddings.append({
                "id": doc["id"],
                "embedding": emb,
                "page": doc.get("page"),
                "chunk_id": doc.get("chunk_id"),
                "image_paths": image_paths
            })

        print(f"✅ Generated embeddings for {len(docs_embeddings)} documents")

        vector_db = VectorDB()
        vector_db.add_embeddings(docs_embeddings)
        vector_db.persist()
            


    def save_embeddings(self, output_path="data/attention_is_all_you_need/docs_embeddings.npy"):
        """
        Save all embeddings to a .npy file for later use.
        """
        embeddings_data = self.generate()
        # Filter out None embeddings
        valid = [d["embedding"] for d in embeddings_data if d["embedding"] is not None]
        all_embeddings = np.stack(valid, axis=0)
        np.save(output_path, all_embeddings)
        print(f"✅ Saved embeddings to {output_path}")
