from fastapi import FastAPI
from app.routers.read_ingest import router as read_ingest_router
from app.routers.query_retriever import router as query_retriever_router


app = FastAPI()

app.include_router(read_ingest_router)
app.include_router(query_retriever_router)

@app.get("/")
def root():
    return {"message": "🚀 RAG API is running"}


