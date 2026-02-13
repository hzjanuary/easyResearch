import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.embedder import embedding_model 

load_dotenv()

# Cấu hình Chroma Path
CHROMA_DIR = "database/chroma_db"

# Prompt đa ngôn ngữ (Giữ nguyên)
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer the user's question based ONLY on the provided context below. "
        "If the answer is not in the context, simply say you don't know in the user's language. "
        "Do not make up information. "
        "\n\nIMPORTANT: Detect the language of the user's question (Vietnamese or English) and answer in that SAME language."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

# --- SỬA ĐỔI: Thêm tham số user_api_key ---
def query_rag_system(question: str, collection_name: str, k_target: int = 10, user_api_key: str = None):
    """
    Hàm xử lý RAG với API Key động.
    """
    
    # 1. Xác định dùng Key nào (Của user hay của hệ thống)
    system_key = os.getenv("GROQ_API_KEY")
    final_api_key = user_api_key if user_api_key and user_api_key.strip() else system_key
    
    if not final_api_key:
        return {
            "answer": "❌ Lỗi: Thiếu API Key. Vui lòng nhập Groq API Key trong cài đặt hoặc thiết lập file .env",
            "sources": []
        }

    # 2. Khởi tạo LLM Dynamic (Mỗi lần hỏi sẽ tạo mới với key đúng)
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            api_key=final_api_key
        )
    except Exception as e:
        return {"answer": f"Lỗi khởi tạo LLM (Kiểm tra lại Key): {str(e)}", "sources": []}

    # 3. Kết nối DB
    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model 
    )

    # 4. Tìm kiếm (Retrieval)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_target,
            "fetch_k": k_target * 3,
            "lambda_mult": 0.7 
        }
    )

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "Tôi không tìm thấy thông tin này trong tài liệu (No relevant documents found).",
            "sources": [],
            "raw_docs": []
        }

    # 5. Ghép Context & Trả lời
    context_text = "\n\n".join(d.page_content for d in docs)
    messages = rag_prompt.format_messages(context=context_text, question=question)
    
    try:
        response = llm.invoke(messages)
        answer_text = response.content.strip()
    except Exception as e:
        answer_text = f"❌ Lỗi khi gọi Groq API: {str(e)}"

    source_names = list(set([d.metadata.get("source", "Unknown") for d in docs]))

    return {
        "answer": answer_text,
        "sources": source_names,
        "raw_docs": [d.page_content for d in docs]
    }