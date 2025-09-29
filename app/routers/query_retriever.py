from fastapi import Query, APIRouter
from app.src.retrieval_mechanism import Retriever
from app.src.store_vector_database import VectorDB
from app.src.generation_model_integration import Generator

router = APIRouter(prefix="/Question", tags=["Query"])

#asking the question
@router.get("/query")
def query_rag_system(question: str = Query(..., description="The question to ask the RAG system")):
    vector_db = VectorDB()
    retriever = Retriever(vector_db=vector_db)
    generator = Generator()

    results = retriever.retrieve(query=question, top_k=5)

    # Separate text and image paths
    text_segments = [d['metadata'].get('text', '') for d in results if d['metadata'].get('text')]
    image_paths = [d['metadata'].get('image_paths', '') for d in results if d['metadata'].get('image_paths')]

    # Generate content
    generated_text = generator.generate_from_text(text_segments)
    generated_image_text = generator.generate_from_images(image_paths)
    return {
        "question": question,
        "retrieved_docs": results,
        "generated_text": generated_text,
        "generated_image_text": generated_image_text
    }