import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def generate_notebook_summary(chunks, api_key=None, llm_provider="groq"):
    """Generate a summary of notebook content from initial chunks."""
    sample_context = "\n\n".join([chunk.page_content for chunk in chunks[:10]])

    if llm_provider == "gemini":
        final_api_key = api_key if api_key else os.getenv("GOOGLE_API_KEY")
        if not final_api_key:
            return "⚠️ Missing Google Gemini API Key for summarization."
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=final_api_key
        )
    else:
        final_api_key = api_key if api_key else os.getenv("GROQ_API_KEY")
        if not final_api_key:
            return "⚠️ Missing API Key for summarization."
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
        return f"❌ Summary error: {str(e)}"