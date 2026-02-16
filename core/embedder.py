import os
import shutil
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import time

CHROMA_DIR = "database/chroma_db"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 EasyResearch running on: {DEVICE.upper()}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

def add_to_vector_db(chunks, collection_name="default_notebook"):
    """Add chunks to ChromaDB collection."""
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.id for chunk in chunks]

    BATCH_SIZE = 500
    total_chunks = len(chunks)
    
    print(f"📥 Ingesting {total_chunks} chunks into '{collection_name}'...")
    
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
        print(f"   ✅ Processed batch {i} -> {end}")
        
    return db

def get_retriever(collection_name="default_notebook"):
    """Get retriever for the RAG pipeline."""
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20}
    )

# ---------------------------------------------------------
# Notebook management
# ---------------------------------------------------------

def get_notebook_stats(notebook_name):
    """Get detailed stats for a notebook: chunk count, source files, storage size."""
    stats = {
        "chunks": 0,
        "files": [],
        "size_mb": 0.0
    }
    
    try:
        if not os.path.exists(CHROMA_DIR):
            return stats
            
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        target_collection = None
        for col in client.list_collections():
            if col.name == notebook_name:
                target_collection = col
                break
        
        if not target_collection:
            return stats
        
        collection = client.get_collection(notebook_name)
        
        stats["chunks"] = collection.count()
        
        if stats["chunks"] > 0:
            result = collection.get(include=["metadatas"])
            if result and result["metadatas"]:
                sources = set()
                for meta in result["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
                stats["files"] = list(sources)
        
        # Calculate directory size
        collection_uuid = str(target_collection.id)
        dir_path = os.path.join(CHROMA_DIR, collection_uuid)
        if os.path.exists(dir_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            stats["size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats
        
    except Exception as e:
        print(f"⚠️ Error getting notebook stats for {notebook_name}: {e}")
        return stats

def get_total_db_size():
    """Get total database size in MB."""
    try:
        if not os.path.exists(CHROMA_DIR):
            return 0.0
            
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(CHROMA_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return round(total_size / (1024 * 1024), 2)
    except Exception as e:
        print(f"⚠️ Error calculating DB size: {e}")
        return 0.0

def get_all_notebooks():
    """Get list of all notebook (collection) names in the database."""
    try:
        if not os.path.exists(CHROMA_DIR):
            return []
            
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        print(f"⚠️ Error listing notebooks: {e}")
        return []

def delete_file_from_notebook(notebook_name, source_name):
    """Delete all chunks of a specific file from a ChromaDB collection by metadata source."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(notebook_name)

        result = collection.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            if meta and meta.get("source") == source_name:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            BATCH = 500
            for i in range(0, len(ids_to_delete), BATCH):
                collection.delete(ids=ids_to_delete[i:i + BATCH])
            print(f"🗑️ Deleted {len(ids_to_delete)} chunks of '{source_name}' from '{notebook_name}'")

        return len(ids_to_delete)
    except Exception as e:
        print(f"❌ Error deleting file {source_name}: {e}")
        return 0


def delete_notebook(notebook_name):
    """Delete a notebook entirely: remove from DB and delete physical directory."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Find collection UUID before deleting
        target_collection = None
        for col in client.list_collections():
            if col.name == notebook_name:
                target_collection = col
                break
        
        collection_uuid = None
        if target_collection:
            collection_uuid = str(target_collection.id)
            print(f"🔍 Found physical directory: {collection_uuid}")

        # Remove from DB (SQLite)
        client.delete_collection(notebook_name)
        print(f"🗑️ Deleted collection from DB: {notebook_name}")
        
        # Remove physical directory
        if collection_uuid:
            dir_path = os.path.join(CHROMA_DIR, collection_uuid)
            if os.path.exists(dir_path):
                try:
                    time.sleep(0.5) 
                    shutil.rmtree(dir_path)
                    print(f"📂 Cleaned up directory: {dir_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete folder immediately (file lock): {e}")

        return True
    except Exception as e:
        print(f"❌ Error deleting notebook {notebook_name}: {e}")
        return False