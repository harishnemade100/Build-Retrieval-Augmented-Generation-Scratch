import os
import json
import requests
import fitz
from app.src.generate_embeddings import EmbeddingsGenerator


class ReadFile:
    """
    A class to handle reading and downloading files.
    """

    def __init__(self, file_path: str = "data/attention_is_all_you_need"):
        self.save_path = file_path
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
        """Fast paragraph-based chunking"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    @staticmethod
    def download_pdf(pdf_url: str, save_path: str) -> None:
        """
        Downloads a PDF from the given URL and saves it to the specified path.
        """
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved PDF: {save_path}")

    def extract_pages_and_images(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        pages = []
        for pageno in range(len(doc)):
            page = doc[pageno]
            text = page.get_text("text")

            # extract images
            image_list = page.get_images(full=True)
            images = []
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_name = f"page_{pageno+1}_img_{img_index}.{image_ext}"
                img_path = os.path.join(self.save_path, img_name)
                with open(img_path, "wb") as imgf:
                    imgf.write(image_bytes)
                images.append(img_path)

            pages.append({"page": pageno+1, "text": text, "images": images})
        return pages

    def ingest(self, pdf_url: str) -> list[dict]:
        pdf_path = os.path.join(self.save_path, "paper.pdf")
        if not os.path.exists(pdf_path):
            print("📥 Downloading PDF...")
            self.download_pdf(pdf_url, pdf_path)

        pages = self.extract_pages_and_images(pdf_path)
        docs = []

        for page in pages:
            page_num = page["page"]
            text_chunks = self.chunk_text(page["text"])

            # Add text chunks
            for i, chunk in enumerate(text_chunks):
                docs.append({
                    "id": f"page{page_num}_chunk{i}",
                    "page": page_num,
                    "chunk_id": i,
                    "text": chunk,
                    "image_paths": []
                })

            # Add images as separate docs
            for img_path in page["images"]:
                docs.append({
                    "id": f"page{page_num}_img_{os.path.basename(img_path)}",
                    "page": page_num,
                    "chunk_id": None,
                    "text": "",
                    "image_paths": [img_path]
                })

        # Save metadata
        meta_path = os.path.join(self.save_path, "docs_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)
        print(f"📑 Ingested {len(docs)} docs, saved metadata: {meta_path}")


        embeddings_generator = EmbeddingsGenerator(metadata_path=meta_path)
        embeddings_generator.generate()
        embeddings_generator.save_embeddings()

