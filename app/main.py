from fastapi import FastAPI, Query
from app.src.read_ingest import ReadFile

app = FastAPI()


@app.get("/")
def root():
    return {"message": "🚀 RAG API is running"}

@app.get("/read-and-ingest")
def read_and_ingest(pdf_url: str = Query(..., description="URL of the PDF to process")):
    reader = ReadFile()
    docs = reader.ingest(pdf_url)
    return {
        "message": "✅ PDF processed and ingested successfully",
        "documents": docs
    }
