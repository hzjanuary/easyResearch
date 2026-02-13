import streamlit as st
import os
import time

# Import các module từ bộ não Core
from core.loader import load_and_split_document
from core.embedder import add_to_vector_db, get_all_notebooks, delete_notebook, get_notebook_stats, get_total_db_size
from core.generator import query_rag_system
from core.summarizer import generate_notebook_summary

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

# CSS tối ưu giao diện
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        background: linear-gradient(90deg, #00ffa3, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 163, 0.3);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00ffa3, #00d4ff);
        color: #1a1a2e;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #ff4757;
        color: white;
    }
    
    /* Project badge */
    .project-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
    }
    
    /* Chat header */
    .chat-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border-left: 4px solid #00ffa3;
    }
    .chat-header h2 {
        margin: 0;
        color: #fff;
    }
    .chat-header p {
        margin: 0.5rem 0 0 0;
        color: #888;
        font-size: 0.9rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00ffa3;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1e2e;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 255, 163, 0.1);
        border-left: 3px solid #00ffa3;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin: 10px 0;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ffa3;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar: Quản lý & Cấu hình
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 easyResearch")
    
    # --- PHẦN 1: QUẢN LÝ DỰ ÁN ---
    with st.container():
        st.markdown("#### 📂 Dự án")
        existing_notebooks = get_all_notebooks()
        total_db_size = get_total_db_size()
        
        # Hiển thị tổng quan Database
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-number">{len(existing_notebooks)}</div>
                <div class="stat-label">Dự án</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{total_db_size}</div>
                <div class="stat-label">MB tổng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        options = ["➕ Tạo mới..."] + existing_notebooks
        selected_option = st.selectbox(
            "Chọn dự án",
            options,
            label_visibility="collapsed",
            help="Chọn dự án để làm việc hoặc tạo mới"
        )
        
        final_notebook_name = "Default_Project"
        
        if selected_option == "➕ Tạo mới...":
            new_name = st.text_input(
                "Tên dự án",
                "My_New_Project",
                label_visibility="collapsed",
                placeholder="Nhập tên dự án..."
            )
            final_notebook_name = new_name.replace(" ", "_").strip()
            st.caption(f"📁 Sẽ tạo: **{final_notebook_name}**")
        else:
            final_notebook_name = selected_option
            st.markdown(f'<div class="project-badge">📖 {final_notebook_name}</div>', unsafe_allow_html=True)
            
            if st.button("🗑️ Xóa dự án này", key="del_btn", type="secondary", use_container_width=True):
                if delete_notebook(final_notebook_name):
                    # Xóa file summary nếu có
                    summary_path = f"database/chroma_db/{final_notebook_name}_summary.txt"
                    if os.path.exists(summary_path):
                        os.remove(summary_path)
                    st.success("✅ Đã xóa thành công!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Không thể xóa!")
        
        # --- HIỂN THỊ TÓM TẮT DỰ ÁN ---
        summary_file = f"database/chroma_db/{final_notebook_name}_summary.txt"
        if os.path.exists(summary_file):
            with st.expander("📄 Tóm tắt Dự án", expanded=False):
                with open(summary_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
        
        # --- DASHBOARD THỐNG KÊ DỰ ÁN ---
        if selected_option != "➕ Tạo mới...":
            with st.expander("📊 Thống kê dự án", expanded=False):
                stats = get_notebook_stats(final_notebook_name)
                
                # Hiển thị thống kê dạng card
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="📄 Đoạn văn",
                        value=stats["chunks"],
                        help="Số lượng chunks trong DB"
                    )
                with col2:
                    st.metric(
                        label="📁 File nguồn",
                        value=len(stats["files"]),
                        help="Số tài liệu đã nạp"
                    )
                
                st.metric(
                    label="💾 Dung lượng",
                    value=f"{stats['size_mb']} MB",
                    help="Dung lượng trên ổ cứng"
                )
                
                # Danh sách file nguồn
                if stats["files"]:
                    st.markdown("**Danh sách tài liệu:**")
                    for i, f in enumerate(stats["files"], 1):
                        st.caption(f"{i}. 📄 {f}")
    
    st.divider()
    
    # --- PHẦN 2: CẤU HÌNH AI ---
    with st.expander("⚙️ Cấu hình", expanded=False):
        user_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Để trống = dùng key mặc định"
        )
        
        search_k = st.slider(
            "🔍 Độ sâu tìm kiếm",
            min_value=3,
            max_value=20,
            value=10,
            help="Số lượng đoạn văn tham khảo"
        )
        
        st.caption(f"Tìm **{search_k}** đoạn văn liên quan nhất")
    
    # --- PHẦN 3: NẠP TÀI LIỆU ---
    with st.expander("📥 Nạp tài liệu", expanded=True):
        uploaded_files = st.file_uploader(
            "Kéo thả hoặc chọn file",
            type=["pdf", "txt", "docx", "py"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Hỗ trợ: PDF, DOCX, TXT, Python"
        )
        
        if uploaded_files:
            st.caption(f"📎 Đã chọn **{len(uploaded_files)}** file")
            
        col1, col2 = st.columns([3, 1])
        with col1:
            process_btn = st.button("🚀 Xử lý", type="primary", use_container_width=True)
        with col2:
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} 📄")
        
        if process_btn:
            if not uploaded_files:
                st.warning("⚠️ Chưa chọn file!")
            else:
                progress_bar = st.progress(0, text="Đang xử lý...")
                for i, uploaded_file in enumerate(uploaded_files):
                    temp_path = f"uploads/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    progress_bar.progress(
                        (i + 0.5) / len(uploaded_files),
                        text=f"📄 {uploaded_file.name[:20]}..."
                    )
                    
                    try:
                        chunks = load_and_split_document(temp_path)
                        add_to_vector_db(chunks, collection_name=final_notebook_name)
                        
                        # Lưu chunks để tạo summary sau
                        if "all_chunks" not in st.session_state:
                            st.session_state.all_chunks = []
                        st.session_state.all_chunks.extend(chunks)
                        
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # --- TỰ ĐỘNG TÓM TẮT SAU KHI NẠP XONG ---
                progress_bar.progress(1.0, text="📝 Đang tạo tóm tắt...")
                try:
                    if "all_chunks" in st.session_state and st.session_state.all_chunks:
                        summary = generate_notebook_summary(st.session_state.all_chunks, api_key=user_key)
                        summary_path = f"database/chroma_db/{final_notebook_name}_summary.txt"
                        with open(summary_path, "w", encoding="utf-8") as f:
                            f.write(summary)
                        st.session_state.all_chunks = []  # Reset
                except Exception as e:
                    st.warning(f"⚠️ Không thể tạo tóm tắt: {e}")
                
                progress_bar.progress(1.0, text="✅ Hoàn tất!")
                time.sleep(1)
                st.rerun()

    st.divider()
    
    # --- NÚT XÓA CHAT ---
    if st.button("🧹 Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Đã xóa lịch sử. Tôi có thể giúp gì?"}]
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("Made with ❤️ by easyResearch")

# ---------------------------------------------------------
# 3. Giao diện Chat
# ---------------------------------------------------------

# Header với thông tin dự án
st.markdown(f"""
<div class="chat-header">
    <h2>💬 Trò chuyện</h2>
    <p>Đang làm việc với: <strong>{final_notebook_name}</strong></p>
</div>
""", unsafe_allow_html=True)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Xin chào! Tôi là trợ lý nghiên cứu AI.\n\n**Bắt đầu bằng cách:**\n1. Chọn hoặc tạo dự án ở sidebar\n2. Nạp tài liệu của bạn\n3. Đặt câu hỏi cho tôi!"}]

if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = final_notebook_name
elif st.session_state.current_notebook != final_notebook_name:
    st.session_state.messages = [{"role": "assistant", "content": f"📂 Đã chuyển sang dự án **{final_notebook_name}**.\n\nHãy đặt câu hỏi về tài liệu trong dự án này!"}]
    st.session_state.current_notebook = final_notebook_name

# Hiển thị chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💭 Đặt câu hỏi về tài liệu của bạn..."):
    # Thêm tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Xử lý và trả lời
    with st.chat_message("assistant", avatar="🧠"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔍 Đang tìm kiếm trong tài liệu..."):
            try:
                result = query_rag_system(
                    prompt,
                    collection_name=final_notebook_name,
                    chat_history=st.session_state.messages,  # Truyền lịch sử chat
                    k_target=search_k,
                    user_api_key=user_key
                )
                
                answer = result["answer"]
                sources = result["sources"]
                
                # Hiệu ứng đánh máy
                words = answer.split()
                for i, word in enumerate(words):
                    full_response += word + " "
                    if i % 3 == 0:  # Cập nhật mỗi 3 từ để mượt hơn
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
                
                message_placeholder.markdown(full_response)
                
                # Hiển thị nguồn tham khảo
                if sources:
                    st.markdown("---")
                    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} tài liệu)", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"{i}. 📄 `{src}`")

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                full_response = "Đã xảy ra lỗi. Vui lòng thử lại."
                message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})