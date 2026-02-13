import os
import shutil
import chromadb # Thư viện cốt lõi để quản lý DB (List/Delete)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import time

# Cấu hình thư mục lưu trữ DB
CHROMA_DIR = "database/chroma_db"

# Tối ưu hóa cho RTX 3050
# Kiểm tra xem có GPU không, nếu có dùng 'cuda', không thì 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 EasyResearch đang chạy trên thiết bị: {DEVICE.upper()}")

# Khởi tạo mô hình Embedding
# Sử dụng model hỗ trợ đa ngôn ngữ (bao gồm Tiếng Việt và Tiếng Anh)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': DEVICE}, # Quan trọng: Đẩy model vào GPU
    encode_kwargs={'normalize_embeddings': True}
)

def add_to_vector_db(chunks, collection_name="default_notebook"):
    """
    Thêm chunks vào ChromaDB theo collection (Notebook) cụ thể.
    """
    # Khởi tạo kết nối tới Chroma
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )
    
    # Tách riêng Texts, Metadatas và IDs để nạp vào DB
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.id for chunk in chunks] # Dùng ID từ hàm hash

    # Xử lý theo Batch (Lô) để tránh tràn VRAM của RTX 3050 (4GB)
    BATCH_SIZE = 500 
    total_chunks = len(chunks)
    
    print(f"📥 Đang nạp {total_chunks} đoạn văn vào Notebook '{collection_name}'...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total_chunks)
        batch_texts = texts[i:end]
        batch_metadatas = metadatas[i:end]
        batch_ids = ids[i:end]
        
        db.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids 
        )
        print(f"   ✅ Đã xử lý batch {i} -> {end}")
        
    return db

def get_retriever(collection_name="default_notebook"):
    """
    Hàm lấy công cụ tìm kiếm cho Generator
    """
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )
    # Sử dụng MMR như dự án gốc để tăng độ đa dạng
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20}
    )

# ---------------------------------------------------------
# CÁC HÀM QUẢN LÝ NOTEBOOK (ĐÃ CẬP NHẬT LOGIC XÓA FOLDER)
# ---------------------------------------------------------

def get_all_notebooks():
    """
    Lấy danh sách tất cả các Notebook (Collection) đang có trong Database
    """
    try:
        # Nếu thư mục chưa tồn tại thì chưa có notebook nào
        if not os.path.exists(CHROMA_DIR):
            return []
            
        # Kết nối trực tiếp vào DB để xem danh sách
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        # Trả về danh sách tên các notebook
        return [c.name for c in collections]
    except Exception as e:
        print(f"⚠️ Lỗi khi lấy danh sách Notebook: {e}")
        return []

def delete_notebook(notebook_name):
    """
    Xóa hoàn toàn một Notebook khỏi Database VÀ Xóa thư mục vật lý trên ổ cứng
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # --- BƯỚC 1: Tìm UUID của thư mục trước khi xóa ---
        target_collection = None
        # Duyệt qua danh sách để tìm đúng collection object
        for col in client.list_collections():
            if col.name == notebook_name:
                target_collection = col
                break
        
        collection_uuid = None
        if target_collection:
            collection_uuid = str(target_collection.id) # Lấy ID thư mục (Ví dụ: 93f17d...)
            print(f"🔍 Đã tìm thấy thư mục vật lý: {collection_uuid}")
        # -------------------------------------------------------

        # --- BƯỚC 2: Xóa khỏi Logic (SQLite) ---
        client.delete_collection(notebook_name)
        print(f"🗑️ Đã xóa Collection khỏi DB: {notebook_name}")
        
        # --- BƯỚC 3: Xóa thư mục vật lý (Ổ cứng) ---
        if collection_uuid:
            dir_path = os.path.join(CHROMA_DIR, collection_uuid)
            if os.path.exists(dir_path):
                try:
                    # Chờ 1 chút để Window nhả file ra (Fix lỗi PermissionError)
                    time.sleep(0.5) 
                    shutil.rmtree(dir_path) # Lệnh xóa ép buộc folder
                    print(f"📂 Đã xóa sạch thư mục rác: {dir_path}")
                except Exception as e:
                    print(f"⚠️ Không thể xóa folder ngay lập tức (Windows đang khóa): {e}")
                    # Nếu không xóa được ngay, nó sẽ thành file rác, lần sau khởi động lại máy xóa cũng được.

        return True
    except Exception as e:
        print(f"❌ Lỗi khi xóa notebook {notebook_name}: {e}")
        return False