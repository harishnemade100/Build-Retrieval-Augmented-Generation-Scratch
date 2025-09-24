import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    """
    Generate embeddings for text and images using the official `jina-clip-v1` model.
    """

    def __init__(self, metadata_path="data/attention_is_all_you_need/docs_metadata.json"):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{metadata_path} not found! Run ingestion first.")

        self.metadata_path = metadata_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.docs_metadata = json.load(f)

        # Load the Jina-CLIP-v1 model
        self.model = SentenceTransformer("jinaai/jina-clip-v1", trust_remote_code=True)

    def generate(self):
        """
        Generate embeddings for all text and image chunks.
        Returns a list of dicts with embeddings.
        """
        docs_embeddings = []

        for doc in tqdm(self.docs_metadata, desc="Generating embeddings"):
            text = doc.get("text", "")
            image_paths = doc.get("image_paths", [])

            embedding = None

            # Encode text
            if text:
                embedding = self.model.encode([text])[0]

            # Encode images
            if image_paths:
                image_embeddings = []
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert("RGB")
                        img_emb = self.model.encode([image])[0]
                        image_embeddings.append(img_emb)

                if image_embeddings:
                    img_mean = np.mean(image_embeddings, axis=0)
                    if embedding is not None:
                        # Combine text + image embeddings by averaging
                        embedding = (embedding + img_mean) / 2
                    else:
                        embedding = img_mean

            docs_embeddings.append({
                "id": doc["id"],
                "embedding": embedding,
                "page": doc.get("page"),
                "chunk_id": doc.get("chunk_id"),
                "image_paths": image_paths
            })

        print(f"✅ Generated embeddings for {len(docs_embeddings)} documents")
        return docs_embeddings

    def save_embeddings(self, output_path="data/attention_is_all_you_need/docs_embeddings.npy"):
        """
        Save all embeddings to a .npy file for later use.
        """
        embeddings_data = self.generate()
        all_embeddings = np.stack([d["embedding"] for d in embeddings_data if d["embedding"] is not None])
        np.save(output_path, all_embeddings)
        print(f"✅ Saved embeddings to {output_path}")
