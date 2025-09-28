# 🧠 Build Retrieval Augmented Generation (RAG) from Scratch

## 📌 Objective
This project demonstrates how to implement a **multi-modal Retrieval Augmented Generation (RAG) system** from scratch using:

- **PHI-3 Vision Model** (for generation)
- **Jina-CLIP-V1** (for text + image embeddings)
- **ChromaDB** (for vector database storage and retrieval)

We apply this system to the research paper **["Attention is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)** to retrieve content and generate responses.

---

## ⚙️ Components

1. **📄 Document Ingestion**  
   - Load PDF using `fitz` (PyMuPDF).  
   - Extract both **text** and **images** from the paper.  
   - Prepare data into chunks for embedding.

2. **🔍 Embeddings Generation**  
   - Use **Jina-CLIP-V1** from Hugging Face `transformers`.  
   - Encode both **text passages** and **images** into embeddings.  

3. **🗄️ Vector Database (ChromaDB)**  
   - Store embeddings in **ChromaDB**.  
   - Each entry contains:  
     - `id`  
     - `text`  
     - `image` (optional)  
     - `embedding`  

4. **📥 Retrieval Mechanism**  
   - On query, generate query embedding using Jina-CLIP-V1.  
   - Retrieve **top-k similar document segments** from ChromaDB.  

5. **🤖 Generation Model (PHI-3 Vision)**  
   - Pass retrieved context to **PHI-3 Vision model**.  
   - Generate **context-aware text output** (answers, summaries, explanations).  

---

## 🖼️ Workflow Diagram

### 🔹 High-Level Architecture
```mermaid
flowchart TD
    A[User Query] --> B[Encode Query with Jina-CLIP-V1]
    B --> C[ChromaDB Retrieval]
    C -->|Top-k Relevant Segments| D[PHI-3 Vision Model]
    D --> E[Generated Output]
🔹 Multi-Modal RAG Pipeline
mermaid
Copy code
flowchart LR
    P[PDF Ingestion] --> E1[Text Extraction]
    P --> E2[Image Extraction]
    E1 --> G1[Text Embeddings - Jina CLIP]
    E2 --> G2[Image Embeddings - Jina CLIP]
    G1 --> DB[ChromaDB Vector Store]
    G2 --> DB
    Q[User Query] --> Q1[Query Embedding]
    Q1 --> DB
    DB --> R[Retrieve Relevant Chunks]
    R --> M[PHI-3 Vision Model]
    M --> O[Generated Answer]
🛠️ Installation
📋 Requirements
You are only allowed to use the following libraries:

torch

chromadb

numpy

io

fitz (PyMuPDF)

requests

PIL

transformers

📥 Setup
bash
Copy code
# Clone repo
git clone https://github.com/your-username/RAG-from-scratch.git
cd RAG-from-scratch

# Create virtual env
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install allowed libraries
pip install torch chromadb numpy pymupdf requests pillow transformers
▶️ Running the Project
Ingest the Document

bash
Copy code
python ingest.py
Extracts text + images from the PDF

Creates embeddings using Jina-CLIP-V1

Stores them in ChromaDB

Run the RAG System

bash
Copy code
python rag.py --query "Explain the concept of self-attention"
Expected Output

text
Copy code
🔍 Retrieved Context:
"Self-attention allows the model to weigh different parts of the input sequence..."

🤖 Generated Answer:
"Self-attention is a mechanism where each word in a sequence can attend to every other word..."
