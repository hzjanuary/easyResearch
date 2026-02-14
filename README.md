# 🧠 easyResearch - AI Research Assistant

<p align="center">
  <b>Advanced Research Assistant powered by RAG (Retrieval-Augmented Generation)</b>
</p>

---

## 📖 Introduction

**easyResearch** is an AI application that helps you query and ask questions on your own documents. The system uses advanced RAG technology to:

- 📄 Read and analyze documents (PDF, DOCX, TXT, Code)
- 🔍 Hybrid Search (Vector + BM25 keyword search)
- 💬 Answer questions based on document content
- 🌍 Multi-language support (Vietnamese & English)
- 🎯 Cross-Encoder Reranking for better accuracy

## ✨ Features

| Feature                     | Description                                        |
| --------------------------- | -------------------------------------------------- |
| 📂 **Workspace Management** | Organize documents by project/topic separately     |
| 📥 **Multi-format Import**  | Support PDF, DOCX, TXT, Python code                |
| 🧠 **Parent Document**      | Small chunks for search, large chunks for context  |
| ⚡ **GPU Acceleration**     | Optimized for NVIDIA GPU (CUDA)                    |
| 🔑 **Multi-LLM Support**    | Groq (LLaMA 3.3) or Google Gemini                  |
| 🌐 **RESTful API**          | Easy integration via FastAPI                       |
| 🎨 **AnythingLLM Theme**    | Dark zinc UI inspired by AnythingLLM & Gemini      |
| 📊 **Workspace Stats**      | Mini stat cards (docs, vectors, storage size)      |
| 📝 **Auto-Summarizer**      | Automatic summary generation after document upload |
| 🔄 **Smart Context**        | Only contextualize when needed (faster response)   |

## 🏗️ System Architecture

```
easyResearch/
├── app.py              # Streamlit Interface (Web UI)
├── main.py             # FastAPI Server (REST API)
├── core/
│   ├── loader.py       # Parent Document Retrieval & Smart Splitter
│   ├── embedder.py     # Vectorization & ChromaDB Management
│   ├── generator.py    # Advanced RAG Pipeline
│   └── summarizer.py   # Auto-Summarization
├── database/
│   └── chroma_db/      # Vector Database Storage
└── uploads/            # Temporary File Storage
```

### Tech Stack

- **LLM**: Groq (LLaMA 3.3 70B) or Google Gemini 2.0 Flash
- **Embedding**: HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker**: CrossEncoder `ms-marco-MiniLM-L-6-v2`
- **Vector DB**: ChromaDB
- **Keyword Search**: BM25 (rank-bm25)
- **Framework**: LangChain, Streamlit, FastAPI

## 🔬 Advanced RAG Pipeline

```
Question → Smart Contextualization → Vector Search
                                          ↓
                                    BM25 Scoring
                                          ↓
                              Cross-Encoder Reranking
                                          ↓
               Hybrid Score (0.7×Rerank + 0.3×BM25)
                                          ↓
                 Parent Document Retrieval → LLM Answer
```

### Key Optimizations

| Component                   | Benefit                                                 |
| --------------------------- | ------------------------------------------------------- |
| **Hybrid Search**           | Combines semantic + keyword matching                    |
| **Parent Document**         | Small chunks (400) for search, large (2000) for context |
| **Smart Contextualization** | Only calls LLM when pronouns/references detected        |
| **Cross-Encoder**           | Local reranking (no API calls)                          |

## 🚀 Installation

### System Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU

### Installation Steps

1. **Clone repository**

   ```bash
   git clone https://github.com/your-username/easyResearch.git
   cd easyResearch
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**

   Create a `.env` file in the root directory:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_gemini_api_key_here  # Optional
   ```

   > 💡 Get Groq API Key at [console.groq.com](https://console.groq.com)
   > 💡 Get Gemini API Key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## 📖 Usage Guide

### Run Web UI (Streamlit)

```bash
streamlit run app.py
```

Access: `http://localhost:8501`

### Run REST API (FastAPI)

```bash
uvicorn main:app --reload
```

Access Swagger UI: `http://localhost:8000/docs`

## 🔌 API Endpoints

### 1. Question & Answer - `POST /ask`

```json
{
  "question": "Your question here",
  "collection_name": "notebook_name"
}
```

**Response:**

```json
{
  "answer": "AI generated answer",
  "sources": ["file1.pdf", "file2.docx"]
}
```

### 2. Upload Document - `POST /upload`

```bash
curl -X POST "http://localhost:8000/upload?collection_name=my_research" \
  -F "file=@document.pdf"
```

## ⚙️ Advanced Configuration

### Parent Document Chunking

| File Type       | Parent Size | Child Size | Notes                      |
| --------------- | ----------- | ---------- | -------------------------- |
| PDF, DOCX       | 2500        | 500        | Preserve long text context |
| Code (.py, .js) | 1500        | 400        | Split by function/class    |
| JSON, CSV       | 1000        | 300        | Don't split mid-object     |
| Default Text    | 800         | 100        | Balanced                   |

### Search Parameters

- **Hybrid Score**: `0.7 × Rerank + 0.3 × BM25`
- **k**: Number of documents to return (default: 10)
- **Min Score Threshold**: 0.1 (filter low relevance)

## 📁 Workspace Management

- **Create New**: Select "➕ New workspace…" from dropdown and name it
- **Switch**: Select workspace from dropdown — badge shows active workspace
- **Delete Workspace**: Go to ⚙️ Settings tab → "🗑 Delete workspace"
- **Clear Chat**: Go to ⚙️ Settings tab → "🗑 Clear chat"
- **Auto-Summary**: Generated automatically after uploading documents

### Sidebar Layout

| Tab / Section             | Function                                    |
| ------------------------- | ------------------------------------------- |
| 📂 **Workspace Selector** | Select/create workspace with stats cards    |
| 📄 **Documents Tab**      | Upload files, view summary & file list      |
| ⚙️ **Settings Tab**       | LLM provider, API key, search depth, delete |

### UI Theme

The interface uses an **AnythingLLM-inspired dark theme** with a **Gemini-style chat input**:

| Element          | Color     | Description                       |
| ---------------- | --------- | --------------------------------- |
| Sidebar          | `#111111` | Deep dark background              |
| Main area        | `#1c1c1f` | Slightly lighter dark             |
| Inputs / Cards   | `#27272a` | Zinc-800 for form elements        |
| Borders          | `#3f3f46` | Subtle zinc-700 borders           |
| Accent (buttons) | `#4f46e5` | Indigo primary buttons            |
| Chat input       | `#303034` | Unified single-color box          |
| Font             | Inter     | Clean sans-serif via Google Fonts |

> **Note**: CSS targets specific selectors (not `*`) to preserve Streamlit's Material Symbols Rounded icons.

## 🛠️ Troubleshooting

| Issue           | Solution                                   |
| --------------- | ------------------------------------------ |
| Missing API Key | Create `.env` file or enter key in sidebar |
| CUDA Error      | Check NVIDIA driver or run on CPU          |
| VRAM Overflow   | Reduce batch size in `embedder.py`         |
| Slow Response   | Already optimized (1-2 LLM calls only)     |

## 📄 License

MIT License - See [LICENSE](LICENSE) file for more details.

---

<p align="center">
  Made with ❤️ by easyResearch
