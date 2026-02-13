from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os

from core.generator import query_rag_system
from core.loader import load_and_split_document
from core.embedder import add_to_vector_db

app = FastAPI(title="EasyResearch API")

# Định nghĩa thư mục uploads tạm thời
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Model cho tin nhắn trong lịch sử chat
class ChatMessage(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str

# Model nhận dữ liệu từ client
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "default_research"  # Tên notebook mặc định
    chat_history: Optional[List[ChatMessage]] = None  # Lịch sử chat (tùy chọn)
    k_target: int = 10  # Số đoạn văn tìm kiếm
    api_key: Optional[str] = None  # API key tùy chọn


# Endpoint 1: Hỏi đáp (Chat)
@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Gửi câu hỏi và nhận câu trả lời từ RAG.
    Hỗ trợ chat_history để ngữ cảnh hóa câu hỏi.
    """
    try:
        # Chuyển đổi chat_history từ Pydantic model sang dict
        history = None
        if request.chat_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        # Gọi hàm logic từ core/generator.py
        result = query_rag_system(
            request.question, 
            request.collection_name,
            chat_history=history,
            k_target=request.k_target,
            user_api_key=request.api_key
        )
        return result
    except Exception as e:
        # In lỗi ra terminal để debug
        print(f"Lỗi: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Upload & Xử lý File (Dynamic)
@app.post("/upload")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    """
    Upload file -> Lưu tạm -> Cắt nhỏ (Loader) -> Vector hóa (Embedder)
    """
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
        print(f"Lỗi upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chạy server bằng lệnh: uvicorn main:app --reload