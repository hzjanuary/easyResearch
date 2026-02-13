import hashlib
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# =============================================================================
# PARENT DOCUMENT RETRIEVAL CONFIG
# =============================================================================
# Small chunks for precise search, parent chunks for context
PARENT_CHUNK_SIZE = 2000   # Đoạn văn cha (gửi cho AI) - Context rộng
CHILD_CHUNK_SIZE = 400     # Đoạn văn con (để search) - Tìm kiếm chính xác
PARENT_OVERLAP = 200
CHILD_OVERLAP = 50


def get_splitting_strategy(file_path):
    """
    Hàm xác định chiến thuật cắt dựa trên loại file
    Trả về: parent_size, child_size, parent_overlap, child_overlap, separators
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # CHIẾN THUẬT 1: TÀI LIỆU VĂN BẢN (Sách, Báo cáo, Luận văn)
    if ext in ['.pdf', '.docx', '.doc']:
        return 2500, 500, 300, 80, ["\n\n", "\n", ". ", " ", ""]
        
    # CHIẾN THUẬT 2: MÃ NGUỒN (Code)
    elif ext in ['.py', '.js', '.java', '.cpp', '.html']:
        return 1500, 400, 100, 30, [
            "\nclass ", "\ndef ", "\nfunction ",
            "\n\n", "\n", " "
        ]
        
    # CHIẾN THUẬT 3: DỮ LIỆU CẤU TRÚC (JSON, CSV)
    elif ext in ['.json', '.csv', '.xml']:
        return 1000, 300, 50, 0, ["\n", "},", "],", " "]
        
    # CHIẾN THUẬT 4: MẶC ĐỊNH
    else:
        return 2000, 400, 200, 50, ["\n\n", "\n", " ", ""]


def load_and_split_document(file_path, use_parent_retrieval=True):
    """
    Load và split document với hỗ trợ Parent Document Retrieval.
    
    Khi use_parent_retrieval=True:
    - Tạo small chunks (child) để search chính xác
    - Lưu parent content trong metadata để gửi cho AI context rộng hơn
    """
    # 1. Load file
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".py":
        loader = TextLoader(file_path, encoding="utf-8") 
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    
    docs = loader.load()
    filename = os.path.basename(file_path)

    # 2. Lấy chiến thuật cắt
    parent_size, child_size, parent_overlap, child_overlap, separators = get_splitting_strategy(file_path)
    
    print(f"⚙️ Auto-Config cho {ext}: Parent={parent_size}, Child={child_size}")

    if not use_parent_retrieval:
        # Fallback: chỉ dùng single-level chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=separators
        )
        splits = splitter.split_documents(docs)
        
        for i, split in enumerate(splits):
            split.metadata["source"] = filename
            split.metadata["chunk_index"] = i
            split.metadata["parent_content"] = split.page_content  # Parent = bản thân
            unique_string = f"{filename}_{i}"
            split.id = hashlib.sha256(unique_string.encode()).hexdigest()
        
        return splits

    # 3. PARENT DOCUMENT RETRIEVAL
    # Bước 3a: Tạo Parent Chunks (đoạn văn lớn)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=separators
    )
    parent_docs = parent_splitter.split_documents(docs)
    
    # Bước 3b: Tạo Child Chunks từ mỗi Parent
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=separators
    )
    
    all_child_chunks = []
    
    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_content = parent_doc.page_content
        
        # Tạo child chunks từ parent này
        # Phải tạo document giả để split
        from langchain_core.documents import Document
        temp_doc = Document(page_content=parent_content, metadata=parent_doc.metadata.copy())
        child_chunks = child_splitter.split_documents([temp_doc])
        
        # Gắn metadata cho mỗi child chunk
        for child_idx, child_chunk in enumerate(child_chunks):
            child_chunk.metadata["source"] = filename
            child_chunk.metadata["parent_index"] = parent_idx
            child_chunk.metadata["child_index"] = child_idx
            child_chunk.metadata["chunk_index"] = len(all_child_chunks)
            
            # LƯU PARENT CONTENT - Đây là điểm quan trọng!
            # Khi retrieve, sẽ trả về parent_content thay vì child content
            child_chunk.metadata["parent_content"] = parent_content
            child_chunk.metadata["parent_page"] = parent_doc.metadata.get("page", 0)
            
            # Tạo unique ID cho child
            unique_string = f"{filename}_p{parent_idx}_c{child_idx}"
            child_chunk.id = hashlib.sha256(unique_string.encode()).hexdigest()
            
            all_child_chunks.append(child_chunk)
    
    print(f"📄 Đã tạo {len(parent_docs)} parent chunks → {len(all_child_chunks)} child chunks")
    
    return all_child_chunks


def load_document_simple(file_path):
    """
    Load document không cắt - Dùng cho tóm tắt hoặc xử lý đặc biệt
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    
    return loader.load()