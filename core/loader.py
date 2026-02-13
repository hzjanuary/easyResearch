import hashlib
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def get_splitting_strategy(file_path):
    """
    Hàm xác định chiến thuật cắt dựa trên loại file
    Trả về: chunk_size, chunk_overlap, separators
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # CHIẾN THUẬT 1: TÀI LIỆU VĂN BẢN (Sách, Báo cáo, Luận văn)
    # Cần chunk to để không bị đứt mạch văn, giữ được tên tác giả, tiêu đề chương
    if ext in ['.pdf', '.docx', '.doc']:
        return 1200, 250, ["\n\n", "\n", ". ", " ", ""]
        
    # CHIẾN THUẬT 2: MÃ NGUỒN (Code)
    # Cần chunk nhỏ, cắt theo Class/Function để code chạy được
    elif ext in ['.py', '.js', '.java', '.cpp', '.html']:
        return 600, 50, [
            "\nclass ", "\ndef ", "\nfunction ", # Ưu tiên cắt theo hàm/lớp
            "\n\n", "\n", " "
        ]
        
    # CHIẾN THUẬT 3: DỮ LIỆU CẤU TRÚC (JSON, CSV, LOG)
    # Cần cực kỳ chính xác, tránh cắt giữa chừng 1 object
    elif ext in ['.json', '.csv', '.xml']:
        return 500, 0, ["\n", "},", "],", " "]
        
    # CHIẾN THUẬT 4: MẶC ĐỊNH (Text thường)
    else:
        return 800, 100, ["\n\n", "\n", " ", ""]

def load_and_split_document(file_path):
    # 1. Load file
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    # Thêm hỗ trợ load code python
    elif ext == ".py":
        loader = TextLoader(file_path, encoding="utf-8") 
    else:
        # Fallback cho các file lạ
        loader = TextLoader(file_path, encoding="utf-8")
    
    docs = loader.load()

    # 2. Lấy chiến thuật cắt ĐỘNG
    chunk_size, chunk_overlap, separators = get_splitting_strategy(file_path)
    
    print(f"⚙️ Auto-Config cho {ext}: Size={chunk_size}, Overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    splits = splitter.split_documents(docs)
    
    # 3. Tạo ID (Giữ nguyên logic cũ)
    filename = os.path.basename(file_path)
    for i, split in enumerate(splits):
        split.metadata["source"] = filename
        split.metadata["chunk_index"] = i
        unique_string = f"{filename}_{i}"
        split.id = hashlib.sha256(unique_string.encode()).hexdigest()
        
    return splits