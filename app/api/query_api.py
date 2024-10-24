from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.llm import RAGModel
from utils.rag_helpers import RagHelper

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


# Initialize LLM and RAG helper
model = RAGModel()
rag_helper = RagHelper()


@router.post("/query")
def query_db(request: QueryRequest):
    try:
        response = rag_helper.query_model(model, request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear_memory")
def clear_memory():
    try:
        rag_helper.clear_memory()
        return {"message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))