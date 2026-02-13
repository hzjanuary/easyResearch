# 🧠 easyResearch - AI Research Assistant

<p align="center">
  <b>Trợ lý nghiên cứu thông minh sử dụng RAG (Retrieval-Augmented Generation)</b>
</p>

---

## 📖 Giới thiệu

**easyResearch** là ứng dụng AI giúp bạn tra cứu và hỏi đáp trên tài liệu của chính mình. Hệ thống sử dụng công nghệ RAG để:

- 📄 Đọc và phân tích tài liệu (PDF, DOCX, TXT, Code)
- 🔍 Tìm kiếm ngữ nghĩa trong kho dữ liệu
- 💬 Trả lời câu hỏi dựa trên nội dung tài liệu
- 🌍 Hỗ trợ đa ngôn ngữ (Tiếng Việt & Tiếng Anh)

## ✨ Tính năng

| Tính năng                  | Mô tả                                               |
| -------------------------- | --------------------------------------------------- |
| 📂 **Quản lý Notebook**    | Tổ chức tài liệu theo dự án/chủ đề riêng biệt       |
| 📥 **Nạp đa định dạng**    | Hỗ trợ PDF, DOCX, TXT, Python code                  |
| 🧠 **Chunking thông minh** | Tự động điều chỉnh cách cắt tài liệu theo loại file |
| ⚡ **GPU Acceleration**    | Tối ưu cho GPU NVIDIA (CUDA)                        |
| 🔑 **API Key linh hoạt**   | Dùng key riêng hoặc key hệ thống                    |
| 🌐 **RESTful API**         | Tích hợp dễ dàng qua FastAPI                        |

## 🏗️ Kiến trúc hệ thống

```
easyResearch/
├── app.py              # Giao diện Streamlit (Web UI)
├── main.py             # FastAPI Server (REST API)
├── core/
│   ├── loader.py       # Đọc & cắt tài liệu thông minh
│   ├── embedder.py     # Vector hóa & quản lý ChromaDB
│   └── generator.py    # Xử lý RAG & gọi LLM
├── database/
│   └── chroma_db/      # Kho vector database
└── uploads/            # Thư mục lưu file tạm
```

### Công nghệ sử dụng

- **LLM**: Groq API (LLaMA 3.3 70B Versatile)
- **Embedding**: HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB**: ChromaDB
- **Framework**: LangChain, Streamlit, FastAPI

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- NVIDIA GPU với CUDA (khuyến nghị) hoặc CPU

### Các bước cài đặt

1. **Clone repository**

   ```bash
   git clone https://github.com/your-username/easyResearch.git
   cd easyResearch
   ```

2. **Tạo môi trường ảo**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Cài đặt dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Cấu hình API Key**

   Tạo file `.env` trong thư mục gốc:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   > 💡 Lấy API Key miễn phí tại [console.groq.com](https://console.groq.com)

## 📖 Hướng dẫn sử dụng

### Chạy Web UI (Streamlit)

```bash
streamlit run app.py
```

Truy cập: `http://localhost:8501`

### Chạy REST API (FastAPI)

```bash
uvicorn main:app --reload
```

Truy cập Swagger UI: `http://localhost:8000/docs`

## 🔌 API Endpoints

### 1. Hỏi đáp - `POST /ask`

```json
{
  "question": "Câu hỏi của bạn",
  "collection_name": "tên_notebook"
}
```

**Response:**

```json
{
  "answer": "Câu trả lời từ AI",
  "sources": ["file1.pdf", "file2.docx"]
}
```

### 2. Upload tài liệu - `POST /upload`

```bash
curl -X POST "http://localhost:8000/upload?collection_name=my_research" \
  -F "file=@document.pdf"
```

## ⚙️ Cấu hình nâng cao

### Chiến thuật cắt tài liệu (Chunking)

| Loại file       | Chunk Size | Overlap | Ghi chú                  |
| --------------- | ---------- | ------- | ------------------------ |
| PDF, DOCX       | 1200       | 250     | Giữ ngữ cảnh văn bản dài |
| Code (.py, .js) | 600        | 50      | Cắt theo function/class  |
| JSON, CSV       | 500        | 0       | Không cắt giữa object    |
| Text mặc định   | 800        | 100     | Cân bằng                 |

### Tham số tìm kiếm

- **Search Type**: MMR (Maximal Marginal Relevance)
- **k**: Số lượng tài liệu trả về (mặc định: 10)
- **fetch_k**: Số lượng ứng viên ban đầu (k × 3)

## 📁 Quản lý Notebook

- **Tạo mới**: Chọn "➕ Tạo Notebook Mới..." và đặt tên
- **Chuyển đổi**: Chọn notebook từ dropdown
- **Xóa**: Nhấn nút "🗑️ Xóa" khi đang mở notebook

## 🛠️ Troubleshooting

| Vấn đề        | Giải pháp                                   |
| ------------- | ------------------------------------------- |
| Thiếu API Key | Tạo file `.env` hoặc nhập key trong sidebar |
| Lỗi CUDA      | Kiểm tra driver NVIDIA hoặc chạy trên CPU   |
| Tràn VRAM     | Giảm batch size trong `embedder.py`         |

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<p align="center">
  Made with ❤️ for researchers and students
</p>
