# 🧠 easyResearch - AI Research Assistant

<p align="center">
  <b>Intelligent Research Assistant powered by RAG (Retrieval-Augmented Generation)</b>
</p>

---

## 📖 Introduction

**easyResearch** is an AI application that helps you query and ask questions on your own documents. The system uses RAG technology to:

- 📄 Read and analyze documents (PDF, DOCX, TXT, Code)
- 🔍 Semantic search across your data repository
- 💬 Answer questions based on document content
- 🌍 Multi-language support (Vietnamese & English)

## ✨ Features

| Feature                    | Description                                        |
| -------------------------- | -------------------------------------------------- |
| 📂 **Notebook Management** | Organize documents by project/topic separately     |
| 📥 **Multi-format Import** | Support PDF, DOCX, TXT, Python code                |
| 🧠 **Smart Chunking**      | Auto-adjust splitting strategy based on file type  |
| ⚡ **GPU Acceleration**    | Optimized for NVIDIA GPU (CUDA)                    |
| 🔑 **Flexible API Key**    | Use your own key or system default                 |
| 🌐 **RESTful API**         | Easy integration via FastAPI                       |
| 🎨 **Modern UI**           | Gradient UI, collapsible panels, progress tracking |
| 🧹 **Chat Management**     | Clear chat history, user/assistant avatars         |

## 🏗️ System Architecture

```
easyResearch/
├── app.py              # Streamlit Interface (Web UI)
├── main.py             # FastAPI Server (REST API)
├── core/
│   ├── loader.py       # Smart Document Reader & Splitter
│   ├── embedder.py     # Vectorization & ChromaDB Management
│   └── generator.py    # RAG Processing & LLM Calls
├── database/
│   └── chroma_db/      # Vector Database Storage
└── uploads/            # Temporary File Storage
```

### Tech Stack

- **LLM**: Groq API (LLaMA 3.3 70B Versatile)
- **Embedding**: HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB**: ChromaDB
- **Framework**: LangChain, Streamlit, FastAPI

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
   ```

   > 💡 Get a free API Key at [console.groq.com](https://console.groq.com)

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

### Document Chunking Strategy

| File Type       | Chunk Size | Overlap | Notes                      |
| --------------- | ---------- | ------- | -------------------------- |
| PDF, DOCX       | 1200       | 250     | Preserve long text context |
| Code (.py, .js) | 600        | 50      | Split by function/class    |
| JSON, CSV       | 500        | 0       | Don't split mid-object     |
| Default Text    | 800        | 100     | Balanced                   |

### Search Parameters

- **Search Type**: MMR (Maximal Marginal Relevance)
- **k**: Number of documents to return (default: 10)
- **fetch_k**: Initial candidate pool (k × 3)

## 📁 Project Management

- **Create New**: Select "➕ Create new..." from dropdown and name it
- **Switch**: Select project from dropdown - badge shows active project
- **Delete Project**: Click "🗑️ Delete this project" button
- **Clear Chat**: Click "🧹 Clear chat history" to reset conversation

### Sidebar Interface

| Panel                   | Function                                  |
| ----------------------- | ----------------------------------------- |
| 📂 **Project**          | Select/create/delete projects, show stats |
| 📥 **Import Documents** | Upload files with detailed progress bar   |
| ⚙️ **Settings**         | API Key and search depth (collapsible)    |

## 🛠️ Troubleshooting

| Issue           | Solution                                   |
| --------------- | ------------------------------------------ |
| Missing API Key | Create `.env` file or enter key in sidebar |
| CUDA Error      | Check NVIDIA driver or run on CPU          |
| VRAM Overflow   | Reduce batch size in `embedder.py`         |

## 📄 License

MIT License - See [LICENSE](LICENSE) file for more details.

---

<p align="center">
  Made with ❤️ for researchers and students
</p>
