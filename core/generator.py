import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import re
from core.embedder import embedding_model 

load_dotenv()

CHROMA_DIR = "database/chroma_db"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_HISTORY_MESSAGES = 10

# Reranker
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

# =============================================================================
# PROMPTS
# =============================================================================

# Contextualization prompt
contextualize_q_system_prompt = """You are a question reformulation expert. Your task is to reformulate the user's latest question into a standalone question that can be understood WITHOUT the chat history.

RULES:
1. If the question contains pronouns (it, this, that, they, he, she, etc.) or references to previous topics, replace them with the actual terms from chat history.
2. If the question is a follow-up (e.g., "What about...", "And the...", "How about..."), incorporate the previous topic.
3. If the question is already self-contained, return it AS-IS.
4. NEVER answer the question - only reformulate it.
5. Keep the same language as the original question.
6. Be concise but complete.

Examples:
- History: "What is Python?" / Answer: "Python is a programming language"
- Original: "What are its main features?" 
- Reformulated: "What are the main features of Python programming language?"
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "Reformulate this question: {input}"),
])

# Prompt Multi-Query Expansion
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a search query expansion expert. Given a question, generate 3 alternative versions:
1. A more specific/detailed version
2. A broader/general version  
3. A version using different keywords/synonyms

Return ONLY the 3 queries, one per line. No numbering, no explanations.
Keep the same language as the input question."""),
    ("human", "{question}"),
])

# Prompt HyDE - Hypothetical Document Embeddings
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert researcher. Given a question, write a hypothetical document passage that would perfectly answer this question. 
The passage should be:
- 100-200 words
- Written as if from a real academic/professional document
- Factual and informative in tone
- In the same language as the question

Do NOT say "This document answers..." - just write the content directly."""),
    ("human", "{question}"),
])

# Prompt CRAG - Document Relevance Grading
crag_grading_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grading expert. Your task is to assess whether a retrieved document is RELEVANT to the user's question.

A document is RELEVANT if it:
- Contains information that directly answers the question
- Provides context or background needed to understand the answer
- Contains key terms, entities, or concepts from the question

Respond with ONLY one of these three words:
- "RELEVANT" - The document clearly helps answer the question
- "PARTIAL" - The document is somewhat related but not directly helpful
- "IRRELEVANT" - The document does not help answer the question"""),
    ("human", """QUESTION: {question}

DOCUMENT CONTENT:
{document}

GRADE:"""),
])

# RAG prompt with history
rag_prompt_with_history = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI research assistant with access to a document database. 
Answer the user's question based ONLY on the provided context below.

GUIDELINES:
1. If the answer is not in the context, clearly say you don't know - don't make up information.
2. Use the conversation summary to understand the flow of discussion.
3. Be concise but thorough. Use bullet points or numbered lists when appropriate.
4. Cite specific parts of the context when relevant.
5. Detect the language of the user's question and answer in that SAME language.

CONVERSATION SUMMARY (for context):
{conversation_summary}
"""
    ),
    (
        "human",
        "RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"
    )
])

# Prompt RAG không có history
rag_prompt_no_history = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI research assistant with access to a document database.
Answer the user's question based ONLY on the provided context below.

GUIDELINES:
1. If the answer is not in the context, clearly say you don't know - don't make up information.
2. Be concise but thorough. Use bullet points or numbered lists when appropriate.
3. Cite specific parts of the context when relevant.
4. Detect the language of the user's question and answer in that SAME language.
"""
    ),
    (
        "human",
        "RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"
    )
])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _tokenize(text: str) -> list:
    """Tokenize text for BM25."""
    return re.findall(r'\w+', text.lower())


def _summarize_conversation(chat_history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    """Summarize conversation history."""
    if not chat_history or len(chat_history) <= 1:
        return "This is the beginning of the conversation."
    
    recent_history = chat_history[-max_messages:]
    summary_parts = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        summary_parts.append(f"- {role}: {content}")
    
    return "\n".join(summary_parts)


def _generate_multi_queries(llm, question: str) -> list:
    """Multi-Query Expansion: generate query variants."""
    try:
        chain = multi_query_prompt | llm
        result = chain.invoke({"question": question})
        queries = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
        return [question] + queries[:3]  # original + 3 variants
    except Exception as e:
        print(f"⚠️ Multi-Query failed: {e}")
        return [question]


def _generate_hyde_document(llm, question: str) -> str:
    """HyDE: Generate hypothetical document for search."""
    try:
        chain = hyde_prompt | llm
        result = chain.invoke({"question": question})
        hyde_doc = result.content.strip()
        print(f"📝 HyDE document generated: {hyde_doc[:100]}...")
        return hyde_doc
    except Exception as e:
        print(f"⚠️ HyDE failed: {e}")
        return question


def _bm25_search(documents: list, query: str, top_k: int = 10) -> list:
    """BM25 Keyword Search."""
    if not documents:
        return []
    
    tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    for i, doc in enumerate(documents):
        doc.metadata["bm25_score"] = float(scores[i])
    
    sorted_docs = sorted(documents, key=lambda x: x.metadata["bm25_score"], reverse=True)
    return sorted_docs[:top_k]


def _crag_grade_documents(llm, question: str, documents: list) -> tuple:
    """CRAG: Grade document relevance. Returns (relevant, partial, irrelevant_count)."""
    relevant_docs = []
    partial_docs = []
    irrelevant_count = 0
    
    grading_chain = crag_grading_prompt | llm
    
    for doc in documents:
        try:
            doc_snippet = doc.page_content[:500]
            result = grading_chain.invoke({
                "question": question,
                "document": doc_snippet
            })
            grade = result.content.strip().upper()
            
            if "RELEVANT" in grade and "IRRELEVANT" not in grade:
                doc.metadata["crag_grade"] = "RELEVANT"
                relevant_docs.append(doc)
            elif "PARTIAL" in grade:
                doc.metadata["crag_grade"] = "PARTIAL"
                partial_docs.append(doc)
            else:
                doc.metadata["crag_grade"] = "IRRELEVANT"
                irrelevant_count += 1
                
        except Exception as e:
            doc.metadata["crag_grade"] = "PARTIAL"
            partial_docs.append(doc)
    
    return relevant_docs, partial_docs, irrelevant_count


def _hybrid_search(db, query: str, hyde_query: str, k_per_method: int = 10) -> list:
    """Hybrid Search: Vector Search + BM25."""
    all_docs = []
    seen_contents = set()
    
    # 1. Vector Search with original query
    vector_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_per_method}
    )
    vector_docs = vector_retriever.invoke(query)
    for doc in vector_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            doc.metadata["retrieval_method"] = "vector"
            all_docs.append(doc)
    
    # 2. Vector Search with HyDE document
    hyde_docs = vector_retriever.invoke(hyde_query)
    for doc in hyde_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            doc.metadata["retrieval_method"] = "hyde"
            all_docs.append(doc)
    
    print(f"📊 Hybrid Search: {len(vector_docs)} vector + {len(hyde_docs)} HyDE = {len(all_docs)} unique docs")
    
    # 3. BM25 on retrieved docs (for reranking)
    if all_docs:
        bm25_ranked = _bm25_search(all_docs.copy(), query, top_k=len(all_docs))
        # Merge BM25 scores into original docs
        bm25_scores = {hash(d.page_content[:100]): d.metadata.get("bm25_score", 0) for d in bm25_ranked}
        for doc in all_docs:
            doc_hash = hash(doc.page_content[:100])
            doc.metadata["bm25_score"] = bm25_scores.get(doc_hash, 0)
    
    return all_docs


# =============================================================================
# HELPER: Check if question needs contextualization
# =============================================================================

def _needs_contextualization(question: str) -> bool:
    """Check if question needs contextualization (contains pronouns/references)."""
    # Context indicator patterns
    context_indicators = [
        # English
        r'\b(it|its|this|that|these|those|they|them|their|he|she|him|her)\b',
        r'\b(the same|above|previous|mentioned|said|such)\b',
        r'\b(what about|how about|and the|also the|another)\b',
        # Vietnamese
        r'\b(nó|này|đó|ở trên|như vậy|còn|thế thì|vậy thì)\b',
        r'\b(cái này|cái đó|điều đó|vấn đề này|chúng)\b',
    ]
    
    question_lower = question.lower()
    for pattern in context_indicators:
        if re.search(pattern, question_lower):
            return True
    return False


# =============================================================================
# MAIN RAG FUNCTION
# =============================================================================

def query_rag_system(question: str, collection_name: str, chat_history: list = None, k_target: int = 10, user_api_key: str = None, llm_provider: str = "groq"):
    """
    OPTIMIZED RAG Pipeline:
    - Vector Search + BM25 Hybrid
    - Cross-Encoder Reranking
    - Single LLM call for answer
    
    Supports: Groq (LLaMA 3.3) and Google Gemini
    """
    
    # 1. API KEY & LLM INITIALIZATION
    if llm_provider == "gemini":
        system_key = os.getenv("GOOGLE_API_KEY")
        final_api_key = user_api_key if user_api_key and user_api_key.strip() else system_key
        
        if not final_api_key:
            return {"answer": "❌ Error: Missing Google Gemini API Key.", "sources": []}
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_output_tokens=1024,
                google_api_key=final_api_key
            )
        except Exception as e:
            return {"answer": f"Error initializing Gemini: {str(e)}", "sources": []}
    else:
        # Default: Groq
        system_key = os.getenv("GROQ_API_KEY")
        final_api_key = user_api_key if user_api_key and user_api_key.strip() else system_key
        
        if not final_api_key:
            return {"answer": "❌ Error: Missing Groq API Key.", "sources": []}

        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1024,
                api_key=final_api_key
            )
        except Exception as e:
            return {"answer": f"Error initializing LLM: {str(e)}", "sources": []}

    # 2. CONTEXTUALIZATION (only when history exists and question has pronouns/references)
    standalone_question = question
    has_history = chat_history and len(chat_history) > 1
    
    # Only contextualize if question shows signs of needing it
    need_context = has_history and _needs_contextualization(question)
    
    if need_context:
        try:
            contextualize_chain = contextualize_q_prompt | llm
            recent_history = chat_history[-MAX_HISTORY_MESSAGES:-1] if len(chat_history) > MAX_HISTORY_MESSAGES else chat_history[:-1]
            
            history_langchain = []
            for msg in recent_history:
                if msg["role"] == "user":
                    history_langchain.append(HumanMessage(content=msg["content"]))
                else:
                    history_langchain.append(AIMessage(content=msg["content"]))
            
            standalone_question = contextualize_chain.invoke({
                "chat_history": history_langchain,
                "input": question
            }).content.strip()
            
            print(f"✨ Contextualized: {standalone_question}")
        except Exception as e:
            print(f"⚠️ Contextualization failed: {e}")

    # 3. DATABASE CONNECTION
    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model 
    )

    # 4. VECTOR SEARCH (Single query - faster)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_target * 2}
    )
    
    all_docs = retriever.invoke(standalone_question)
    
    if not all_docs:
        return {
            "answer": "No relevant information found in the documents.",
            "sources": [],
            "raw_docs": [],
            "pipeline_info": {"retrieval": "no_docs_found"}
        }

    # 5. BM25 SCORING (Hybrid component - no LLM)
    bm25_ranked = _bm25_search(all_docs.copy(), standalone_question, top_k=len(all_docs))
    bm25_scores = {hash(d.page_content[:100]): d.metadata.get("bm25_score", 0) for d in bm25_ranked}
    
    # Normalize BM25 scores
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
    for doc in all_docs:
        doc_hash = hash(doc.page_content[:100])
        raw_score = bm25_scores.get(doc_hash, 0)
        doc.metadata["bm25_score"] = raw_score / max_bm25 if max_bm25 > 0 else 0

    # 6. CROSS-ENCODER RERANKING
    pairs = [[standalone_question, doc.page_content] for doc in all_docs]
    rerank_scores = reranker_model.predict(pairs)
    
    for i, doc in enumerate(all_docs):
        doc.metadata["rerank_score"] = float(rerank_scores[i])
        # Hybrid score: 0.7 * rerank + 0.3 * bm25
        doc.metadata["hybrid_score"] = 0.7 * doc.metadata["rerank_score"] + 0.3 * doc.metadata["bm25_score"]
    
    # Filter và sort by hybrid score
    min_score_threshold = 0.1
    filtered_docs = [d for d in all_docs if d.metadata["hybrid_score"] >= min_score_threshold]
    
    if not filtered_docs:
        filtered_docs = sorted(all_docs, key=lambda x: x.metadata["hybrid_score"], reverse=True)[:k_target]
    else:
        filtered_docs = sorted(filtered_docs, key=lambda x: x.metadata["hybrid_score"], reverse=True)[:k_target]
    
    final_docs = filtered_docs
    
    top_scores = [round(d.metadata["hybrid_score"], 2) for d in final_docs[:3]]
    print(f"🎯 Selected {len(final_docs)} docs (hybrid scores: {top_scores}...)")

    if not final_docs:
        return {
            "answer": "No relevant information found in the documents.",
            "sources": [],
            "raw_docs": [],
            "pipeline_info": {"retrieval": "no_relevant_docs"}
        }

    # 7. GENERATE ANSWER (Single LLM call — uses Parent Content)
    context_text = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.metadata.get('parent_content', d.page_content)}" 
        for d in final_docs
    ])
    
    try:
        if has_history:
            conversation_summary = _summarize_conversation(chat_history)
            messages = rag_prompt_with_history.format_messages(
                context=context_text, 
                question=question,
                conversation_summary=conversation_summary
            )
        else:
            messages = rag_prompt_no_history.format_messages(
                context=context_text, 
                question=question
            )
        
        response = llm.invoke(messages)
        answer_text = response.content.strip()
            
    except Exception as e:
        answer_text = f"❌ Error calling API: {str(e)}"

    source_names = list(set([d.metadata.get("source", "Unknown") for d in final_docs]))

    return {
        "answer": answer_text,
        "sources": source_names,
        "raw_docs": [
            f"[Score: {d.metadata.get('hybrid_score', 0):.2f}] {d.page_content[:200]}..." 
            for d in final_docs
        ],
        "standalone_question": standalone_question if standalone_question != question else None,
        "pipeline_info": {
            "total_retrieved": len(all_docs),
            "final_docs": len(final_docs),
            "contextualized": need_context
        }
    }