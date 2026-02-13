import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder # Cần cài: pip install sentence-transformers
from core.embedder import embedding_model 

load_dotenv()

# Cấu hình Chroma Path và Thiết bị
CHROMA_DIR = "database/chroma_db"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Khởi tạo Reranker mô hình MiniLM (Tối ưu cho VRAM 4GB của RTX 3050)
# Mô hình này so sánh trực tiếp Query và Context để chấm điểm độ liên quan
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

# Prompt ngữ cảnh hóa câu hỏi (Contextualization)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

# Prompt đa ngôn ngữ
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer the user's question based ONLY on the provided context below. "
        "If the answer is not in the context, simply say you don't know in the user's language. "
        "\n\nIMPORTANT: Detect the language of the user's question (Vietnamese or English) and answer in that SAME language."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

def query_rag_system(question: str, collection_name: str, chat_history: list = None, k_target: int = 10, user_api_key: str = None):
    """
    Hàm xử lý RAG kết hợp Reranking và Chat History Contextualization.
    """
    
    # 1. Xác định dùng Key nào
    system_key = os.getenv("GROQ_API_KEY")
    final_api_key = user_api_key if user_api_key and user_api_key.strip() else system_key
    
    if not final_api_key:
        return {
            "answer": "❌ Lỗi: Thiếu API Key Groq.",
            "sources": []
        }

    # 2. Khởi tạo LLM Dynamic
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
            api_key=final_api_key
        )
    except Exception as e:
        return {"answer": f"Lỗi khởi tạo LLM: {str(e)}", "sources": []}

    # 3. NGỮ CẢNH HÓA CÂU HỎI (Nếu có lịch sử chat)
    # Viết lại câu hỏi để AI hiểu ngữ cảnh từ cuộc trò chuyện trước đó
    standalone_question = question
    if chat_history and len(chat_history) > 1:  # Cần ít nhất 1 cặp hỏi-đáp
        try:
            contextualize_chain = contextualize_q_prompt | llm
            # Chuyển đổi list messages từ Streamlit sang dạng LangChain message
            history_langchain = []
            for msg in chat_history[:-1]:  # Bỏ tin nhắn cuối (câu hỏi hiện tại)
                if msg["role"] == "user":
                    history_langchain.append(HumanMessage(content=msg["content"]))
                else:
                    history_langchain.append(AIMessage(content=msg["content"]))
            
            # Viết lại câu hỏi
            standalone_question = contextualize_chain.invoke({
                "chat_history": history_langchain,
                "input": question
            }).content
            print(f"🔍 Câu hỏi đã được làm rõ: {standalone_question}")
        except Exception as e:
            print(f"⚠️ Không thể ngữ cảnh hóa câu hỏi: {e}")
            standalone_question = question

    # 4. Kết nối DB
    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model 
    )

    # 5. GIAI ĐOẠN 1: Retrieval (Lấy rộng - k_target * 2)
    # Chúng ta lấy nhiều ứng viên hơn để Reranker có dữ liệu lọc
    # Sử dụng standalone_question (đã được ngữ cảnh hóa) để tìm kiếm chính xác hơn
    retriever = db.as_retriever(
        search_type="similarity", # Dùng similarity để lấy thô nhanh nhất
        search_kwargs={"k": k_target * 2} 
    )
    initial_docs = retriever.invoke(standalone_question)

    if not initial_docs:
        return {
            "answer": "Tôi không tìm thấy thông tin này trong tài liệu.",
            "sources": [],
            "raw_docs": []
        }

    # 6. GIAI ĐOẠN 2: Reranking (Lọc tinh bằng Cross-Encoder)
    # Tạo cặp [Câu hỏi đã ngữ cảnh hóa, Đoạn văn] để Reranker chấm điểm
    pairs = [[standalone_question, doc.page_content] for doc in initial_docs]
    scores = reranker_model.predict(pairs)

    # Gắn điểm số vào metadata và sắp xếp lại
    for i, doc in enumerate(initial_docs):
        doc.metadata["rerank_score"] = float(scores[i])
    
    # Sắp xếp giảm dần theo điểm và lấy ra đúng k_target đoạn tốt nhất
    reranked_docs = sorted(initial_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)[:k_target]

    # 7. Ghép Context & Trả lời
    # LƯU Ý: Dùng câu hỏi gốc (question) để AI trả lời tự nhiên
    context_text = "\n\n".join(d.page_content for d in reranked_docs)
    messages = rag_prompt.format_messages(context=context_text, question=question)
    
    try:
        response = llm.invoke(messages)
        answer_text = response.content.strip()
    except Exception as e:
        answer_text = f"❌ Lỗi khi gọi Groq API: {str(e)}"

    source_names = list(set([d.metadata.get("source", "Unknown") for d in reranked_docs]))

    # Trả về kết quả kèm theo Score để bạn Debug trên giao diện
    return {
        "answer": answer_text,
        "sources": source_names,
        "raw_docs": [f"[Re-rank Score: {d.metadata['rerank_score']:.2f}] {d.page_content}" for d in reranked_docs]
    }