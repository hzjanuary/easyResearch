import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def generate_notebook_summary(chunks, api_key=None):
    """
    Tóm tắt nội dung chính của toàn bộ Notebook dựa trên các đoạn văn đầu tiên.
    """
    final_api_key = api_key if api_key else os.getenv("GROQ_API_KEY")
    if not final_api_key:
        return "⚠️ Không có API Key để tạo tóm tắt."

    # Lấy khoảng 8-10 đoạn văn đầu tiên (thường chứa Introduction/Abstract)
    sample_context = "\n\n".join([chunk.page_content for chunk in chunks[:10]])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=final_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Summarize the following document snippets into a concise overview. "
                   "Focus on: Main Topic, Key Objectives, and Target Audience. "
                   "Answer in the language of the text (Vietnamese or English)."),
        ("human", "Context to summarize:\n{context}")
    ])

    try:
        messages = prompt.format_messages(context=sample_context)
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"❌ Lỗi tóm tắt: {str(e)}"