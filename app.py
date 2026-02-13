import streamlit as st
import os
import time

# Import các module từ bộ não Core
from core.loader import load_and_split_document
from core.embedder import add_to_vector_db, get_all_notebooks, delete_notebook
from core.generator import query_rag_system

# ---------------------------------------------------------
# 1. Cấu hình giao diện Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="easyResearch - AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

# CSS giao diện
st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #1e1e1e;}
    .stButton > button {width: 100%; border-radius: 8px; font-weight: bold;}
    .source-box {
        padding: 12px; background-color: #2b2d30; 
        border-radius: 8px; margin-bottom: 8px; 
        border-left: 4px solid #00ffa3; font-size: 0.9em;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #ff4b4b; color: white; border: none;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar: Quản lý & Cấu hình
# ---------------------------------------------------------
with st.sidebar:
    st.title("📂 easyResearch")
    
    st.divider()
    
    # --- PHẦN 1: QUẢN LÝ NOTEBOOK ---
    st.subheader("1. Chọn Dự án")
    existing_notebooks = get_all_notebooks()
    options = ["➕ Tạo Notebook Mới..."] + existing_notebooks
    selected_option = st.selectbox("Danh sách dự án:", options)
    
    final_notebook_name = "Default_Project"
    
    if selected_option == "➕ Tạo Notebook Mới...":
        new_name = st.text_input("Nhập tên dự án mới:", "My_New_Project")
        final_notebook_name = new_name.replace(" ", "_").strip()
    else:
        final_notebook_name = selected_option
        col1, col2 = st.columns([1, 1])
        with col1: st.info(f"Mở: {final_notebook_name}")
        with col2:
            if st.button("🗑️ Xóa", key="del_btn", type="secondary"):
                if delete_notebook(final_notebook_name):
                    st.success(f"Đã xóa!")
                    time.sleep(1)
                    st.rerun()
                else: st.error("Lỗi xóa!")
    
    st.divider()
    
    # --- PHẦN 2: NẠP DỮ LIỆU ---
    st.subheader("2. Nạp tài liệu")
    uploaded_files = st.file_uploader("Thêm PDF/DOCX/Code", type=["pdf", "txt", "docx", "py"], accept_multiple_files=True)
    
    if st.button("🚀 Xử lý & Học dữ liệu", type="primary"):
        if not uploaded_files: st.warning("Chọn file trước!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = f"uploads/{uploaded_file.name}"
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                status_text.info(f"⏳ Đang đọc: {uploaded_file.name}...")
                try:
                    chunks = load_and_split_document(temp_path)
                    add_to_vector_db(chunks, collection_name=final_notebook_name)
                    os.remove(temp_path)
                except Exception as e: st.error(f"❌ Lỗi: {e}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.success("✅ Hoàn tất!")
            time.sleep(1)
            st.rerun()

    st.divider()

    # --- PHẦN 3: CẤU HÌNH AI (ĐÃ CẬP NHẬT) ---
    st.subheader("3. Cấu hình AI")
    
    # Input nhập API Key
    user_key = st.text_input(
        "Groq API Key (Tùy chọn)", 
        type="password", 
        help="Nhập key của bạn nếu muốn dùng riêng. Để trống sẽ dùng Key mặc định của hệ thống."
    )
    
    search_k = st.slider("Độ sâu tìm kiếm", 3, 20, 10)

# ---------------------------------------------------------
# 3. Giao diện Chat
# ---------------------------------------------------------
st.header(f"💬 Chat: {final_notebook_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì?"}]

if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = final_notebook_name
elif st.session_state.current_notebook != final_notebook_name:
    st.session_state.messages = [{"role": "assistant", "content": f"Đã chuyển sang dự án: {final_notebook_name}."}]
    st.session_state.current_notebook = final_notebook_name

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Đang tra cứu..."):
            try:
                # --- TRUYỀN KEY NGƯỜI DÙNG VÀO HÀM ---
                result = query_rag_system(
                    prompt, 
                    collection_name=final_notebook_name, 
                    k_target=search_k,
                    user_api_key=user_key  # <--- Quan trọng
                )
                
                answer = result["answer"]
                sources = result["sources"]
                
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                if sources:
                    st.markdown("---")
                    with st.expander("📚 Nguồn tham khảo"):
                        for src in sources: st.markdown(f"- 📄 `{src}`")

            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
                full_response = "Lỗi hệ thống."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})