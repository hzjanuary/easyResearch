from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os

from core.generator import query_rag_system
from core.loader import load_and_split_document
from core.embedder import add_to_vector_db

app = FastAPI(title="EasyResearch API")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    collection_name: str = "default_research"
    chat_history: Optional[List[ChatMessage]] = None
    k_target: int = 10
    api_key: Optional[str] = None


# Endpoint 1: Question & Answer
@app.post("/ask")
def ask_question(request: QueryRequest):
    """Send a question and receive a RAG-powered answer."""
    try:
        history = None
        if request.chat_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        result = query_rag_system(
            request.question, 
            request.collection_name,
            chat_history=history,
            k_target=request.k_target,
            user_api_key=request.api_key
        )
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Upload & Process File
@app.post("/upload")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    """Upload file -> Save -> Split (Loader) -> Vectorize (Embedder)"""
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    
    try:

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            

        chunks = load_and_split_document(file_location)
        

        add_to_vector_db(chunks, collection_name)
        

        os.remove(file_location)
        
        return {
            "status": "success", 
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "collection": collection_name
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn main:app --reload