import streamlit as st
import os
import time
import json

# Import các module từ bộ não Core
from core.loader import load_and_split_document
from core.embedder import add_to_vector_db, get_all_notebooks, delete_notebook, delete_file_from_notebook, get_notebook_stats, get_total_db_size
from core.generator import query_rag_system
from core.summarizer import generate_notebook_summary


# ---------------------------------------------------------
# Helper: persistent chat history per workspace
# ---------------------------------------------------------
CHAT_DIR = "database/chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)


def _chat_path(notebook_name):
    return os.path.join(CHAT_DIR, f"{notebook_name}.json")


def save_chat(notebook_name, messages):
    """Save full conversation to disk."""
    with open(_chat_path(notebook_name), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_chat(notebook_name):
    """Load full conversation from disk (or return None)."""
    path = _chat_path(notebook_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def delete_chat(notebook_name):
    """Delete saved conversation."""
    path = _chat_path(notebook_name)
    if os.path.exists(path):
        os.remove(path)


def get_recent_questions(notebook_name, limit=5):
    """Extract recent user questions from saved chat."""
    msgs = load_chat(notebook_name)
    if not msgs:
        return []
    questions = [m["content"] for m in msgs if m["role"] == "user"]
    return questions[-limit:][::-1]  # newest first


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

# =============================================================
# CSS — AnythingLLM-inspired dark theme
# KEY FIX: target font-family on SPECIFIC selectors only,
#          never on *, so Material Symbols Rounded icons work.
# =============================================================
st.markdown(r"""
<style>
/* ── Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply Inter ONLY to content elements, NOT to icon fonts */
body, p, div, h1, h2, h3, h4, h5, h6,
label, input, textarea, a, li, td, th,
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stCaption"] {
    font-family: 'Inter', sans-serif !important;
}

/* Restore Material Symbols for ALL Streamlit icon elements */
span[data-testid],
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] *,
[data-testid="baseButton-headerNoPadding"] *,
[data-testid="baseButton-header"] *,
button[kind="headerNoPadding"] *,
button[kind="header"] *,
.material-symbols-rounded,
[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
}

/* ── Main background ─────────────────────────────────── */
.stApp {
    background-color: #1c1c1f !important;
}

/* ── Top header bar ──────────────────────────────────── */
header[data-testid="stHeader"] {
    background-color: #1c1c1f !important;
    border-bottom: none !important;
}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #2d2d30 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* ── Sidebar title ───────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    margin: 0 0 1.2rem 0 !important;
    padding: 0 !important;
    background: none !important;
    -webkit-text-fill-color: unset !important;
    text-align: center !important;
}

/* ── Sidebar h4 section labels ───────────────────────── */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
    color: #71717a !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 1rem 0 0.4rem 0 !important;
}

/* ── Sidebar captions ────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stCaption"] p {
    color: #52525b !important;
    font-size: 0.75rem !important;
}

/* ── Text inputs ─────────────────────────────────────── */
input[type="text"], input[type="password"], textarea {
    background-color: #27272a !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    color: #e4e4e7 !important;
    caret-color: #e4e4e7 !important;
}
input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* ── Select-boxes ────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background-color: #27272a !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    color: #e4e4e7 !important;
}

/* ── Buttons — primary (indigo) ──────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: background-color 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    background-color: #4338ca !important;
}

/* ── Buttons — secondary / default ───────────────────── */
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {
    background-color: transparent !important;
    color: #a1a1aa !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind]):hover {
    background-color: #27272a !important;
    color: #ffffff !important;
    border-color: #52525b !important;
}

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #3f3f46 !important;
    border-radius: 10px !important;
    background-color: #1f1f23 !important;
    padding: 0.75rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #27272a !important;
    color: #d4d4d8 !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 6px !important;
}

/* ── Expanders ───────────────────────────────────────── */
[data-testid="stExpander"] > details {
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
    background-color: #18181b !important;
}
[data-testid="stExpander"] > details > summary {
    color: #a1a1aa !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stExpander"] > details > summary:hover {
    color: #e4e4e7 !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #27272a !important;
    margin: 0.8rem 0 !important;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1f1f23 !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
}
[data-testid="stMetric"] label {
    color: #71717a !important;
    font-size: 0.75rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e4e4e7 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}

/* ── Progress bar ────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background-color: #6366f1 !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] p {
    text-align: center !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background-color: #1f1f23 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    border: 1px solid #27272a !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: #71717a !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background-color: #27272a !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Chat messages ───────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.8rem 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #d4d4d8 !important;
    line-height: 1.7;
}

/* ── Chat input box ──────────────────────────────────── */
[data-testid="stChatInput"] {
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div,
[data-testid="stChatInput"] div {
    background-color: #303034 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] > div {
    border-radius: 14px !important;
    padding: 6px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #303034 !important;
    color: #e4e4e7 !important;
    padding: 0.45rem 1.2rem !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] button {
    background-color: #303034 !important;
    border: none !important;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #52525b; }

/* ── Custom HTML components ──────────────────────────── */
.workspace-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #27272a;
    border: 1px solid #3f3f46;
    padding: 6px 14px;
    border-radius: 6px;
    color: #e4e4e7;
    font-weight: 500;
    font-size: 0.85rem;
    margin: 4px 0 8px 0;
}

.stats-row {
    display: flex;
    gap: 8px;
    margin: 8px 0;
}
.stat-card {
    flex: 1;
    background-color: #1f1f23;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.stat-card .val {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e4e4e7;
}
.stat-card .lbl {
    font-size: 0.6rem;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

.sidebar-footer {
    color: #3f3f46;
    font-size: 0.7rem;
    text-align: center;
    padding: 0.5rem 0;
}

/* ── Bottom bar: push chat input up to make room for radio ─ */
[data-testid="stBottomBlockContainer"] {
    padding-bottom: 2.6rem !important;
}

/* ── Search mode radio pills (fixed bottom center) ───── */
.st-key-mode_radio {
    position: fixed !important;
    bottom: 0.25rem;
    left: calc(50% + 7rem);
    transform: translateX(-50%);
    z-index: 999;
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-mode_radio [data-testid="stRadio"] > label {
    display: none !important;
}
.st-key-mode_radio [role="radiogroup"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 2px !important;
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 22px;
    padding: 3px 4px;
}
.st-key-mode_radio label[data-baseweb="radio"] {
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 18px;
    transition: all 0.15s ease;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 68px;
    height: 28px;
    cursor: pointer !important;
}
.st-key-mode_radio label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
.st-key-mode_radio label[data-baseweb="radio"] > div:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-mode_radio label[data-baseweb="radio"] p {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #71717a !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    white-space: nowrap !important;
}
.st-key-mode_radio label[data-baseweb="radio"]:hover {
    background-color: rgba(63, 63, 70, 0.5);
}
.st-key-mode_radio label[data-baseweb="radio"]:has(input:checked) {
    background-color: #3f3f46;
}
.st-key-mode_radio label[data-baseweb="radio"]:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* File delete button – compact style */
button[class*="st-key-del_file_"] {
    padding: 0 !important;
    min-height: 0 !important;
    width: 24px !important;
    height: 24px !important;
    font-size: 13px !important;
    background: transparent !important;
    border: 1px solid #3f3f46 !important;
    color: #71717a !important;
    border-radius: 4px !important;
    margin-top: 2px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
button[class*="st-key-del_file_"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
button[class*="st-key-del_file_"]:hover {
    background: #7f1d1d !important;
    border-color: #dc2626 !important;
    color: #fca5a5 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    # Logo
    st.markdown("# 🧠 easyResearch")

    # --- WORKSPACE ---
    st.markdown("#### Workspaces")

    existing_notebooks = get_all_notebooks()

    options = ["➕ New workspace…"] + existing_notebooks
    selected_option = st.selectbox(
        "Workspace", options,
        label_visibility="collapsed",
        index=1 if existing_notebooks else 0,
    )

    final_notebook_name = "Default_Project"

    if selected_option == "➕ New workspace…":
        new_name = st.text_input(
            "Name", "New_Project",
            label_visibility="collapsed",
            placeholder="Enter workspace name…",
        )
        final_notebook_name = new_name.replace(" ", "_").strip()
        st.caption(f"Will create **{final_notebook_name}**")
    else:
        final_notebook_name = selected_option
        st.markdown(
            f'<div class="workspace-badge">📂 {final_notebook_name}</div>',
            unsafe_allow_html=True,
        )

        # Mini stats
        stats = get_notebook_stats(final_notebook_name)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card"><div class="val">{len(stats["files"])}</div><div class="lbl">Docs</div></div>
            <div class="stat-card"><div class="val">{stats["chunks"]}</div><div class="lbl">Vectors</div></div>
            <div class="stat-card"><div class="val">{stats["size_mb"]}</div><div class="lbl">MB</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- ACTIONS (Tabs) ---
    tab_docs, tab_cfg = st.tabs(["📄 Documents", "⚙️ Settings"])

    with tab_docs:
        # Upload
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "txt", "docx", "py"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.caption(f"Selected **{len(uploaded_files)}** file(s)")
            process_btn = st.button("Upload & Process", type="primary", use_container_width=True)
        else:
            process_btn = False

        if process_btn and uploaded_files:
            progress_bar = st.progress(0, text="Processing…")
            for i, uploaded_file in enumerate(uploaded_files):
                temp_path = f"uploads/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    chunks = load_and_split_document(temp_path)
                    add_to_vector_db(chunks, collection_name=final_notebook_name)

                    if "all_chunks" not in st.session_state:
                        st.session_state.all_chunks = []
                    st.session_state.all_chunks.extend(chunks)

                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            # Auto summary
            progress_bar.progress(1.0, text="Generating summary…")
            try:
                if "all_chunks" in st.session_state and st.session_state.all_chunks:
                    summary = generate_notebook_summary(
                        st.session_state.all_chunks,
                        api_key=st.session_state.get("user_api_key", ""),
                        llm_provider=st.session_state.get("llm_provider", "groq"),
                    )
                    summary_path = f"database/chroma_db/{final_notebook_name}_summary.txt"
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                    st.session_state.all_chunks = []
            except Exception:
                pass

            progress_bar.progress(1.0, text="Done!")
            time.sleep(0.8)
            st.rerun()

        # Summary viewer
        summary_file = f"database/chroma_db/{final_notebook_name}_summary.txt"
        if os.path.exists(summary_file):
            with st.expander("📝 Summary", expanded=False):
                with open(summary_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

        # File list
        if selected_option != "➕ New workspace…":
            _stats = get_notebook_stats(final_notebook_name)
            if _stats["files"]:
                with st.expander(f"📁 Files ({len(_stats['files'])})", expanded=False):
                    for i, fname in enumerate(_stats["files"], 1):
                        col_name, col_del = st.columns([0.85, 0.15])
                        display_name = fname if len(fname) <= 30 else fname[:27] + "…"
                        col_name.caption(f"{i}. {display_name}")
                        if col_del.button("✕", key=f"del_file_{hash(fname)}", help=f"Delete {fname}"):
                            deleted = delete_file_from_notebook(final_notebook_name, fname)
                            if deleted:
                                st.toast(f"Deleted {fname} ({deleted} chunks)")
                                time.sleep(0.5)
                                st.rerun()
        # Recent questions
        recent_qs = get_recent_questions(final_notebook_name)
        if recent_qs:
            with st.expander(f"🔍 Recent ({len(recent_qs)})", expanded=False):
                for q in recent_qs:
                    display = q if len(q) <= 35 else q[:35] + "…"
                    if st.button(display, key=f"hist_{hash(q)}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()
    with tab_cfg:
        # LLM Provider
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Groq (LLaMA 3.3 70B)", "Google Gemini"],
        )

        if "Groq" in llm_provider:
            user_key = st.text_input("API Key", type="password", placeholder="gsk_…")
            st.session_state.llm_provider = "groq"
        else:
            user_key = st.text_input("API Key", type="password", placeholder="AIza…")
            st.session_state.llm_provider = "gemini"

        st.session_state.user_api_key = user_key

        st.divider()

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help?"}
            ]
            delete_chat(final_notebook_name)
            st.rerun()

        if selected_option != "➕ New workspace…":
            if st.button("🗑 Delete workspace", type="secondary", use_container_width=True):
                if delete_notebook(final_notebook_name):
                    summary_path = f"database/chroma_db/{final_notebook_name}_summary.txt"
                    if os.path.exists(summary_path):
                        os.remove(summary_path)
                    delete_chat(final_notebook_name)
                    st.success("Deleted!")
                    time.sleep(0.5)
                    st.rerun()

    # Footer
    st.markdown('<div class="sidebar-footer">easyResearch · RAG Assistant</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Main Chat Area
# ---------------------------------------------------------

# Session state — load persisted chat or show welcome
_default_welcome = [
    {
        "role": "assistant",
        "content": (
            "Welcome to your workspace.\n\n"
            "To get started, **upload a document** in the sidebar "
            "or *send a chat*."
        ),
    }
]

if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = final_notebook_name
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else list(_default_welcome)
elif st.session_state.current_notebook != final_notebook_name:
    # Workspace switched — save old, load new
    save_chat(st.session_state.current_notebook, st.session_state.messages)
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else [
        {
            "role": "assistant",
            "content": f"Switched to workspace **{final_notebook_name}**.\n\nAsk me anything about your documents!",
        }
    ]
    st.session_state.current_notebook = final_notebook_name
elif "messages" not in st.session_state:
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else list(_default_welcome)

# Chat history
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Search mode
SEARCH_MODES = {"Fast": 5, "Accurate": 10, "Detailed": 18}

if "search_mode" not in st.session_state:
    st.session_state.search_mode = "Accurate"

# Chat input
prompt = st.chat_input("Send a message")

# Search mode radio pills — rendered below chat input
selected_mode = st.radio(
    "mode", list(SEARCH_MODES.keys()),
    index=list(SEARCH_MODES.keys()).index(st.session_state.search_mode),
    horizontal=True,
    label_visibility="collapsed",
    key="mode_radio",
)
st.session_state.search_mode = selected_mode
search_k = SEARCH_MODES[selected_mode]

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Searching documents…"):
            try:
                result = query_rag_system(
                    prompt,
                    collection_name=final_notebook_name,
                    chat_history=st.session_state.messages,
                    k_target=search_k,
                    user_api_key=user_key,
                    llm_provider=st.session_state.get("llm_provider", "groq"),
                )

                answer = result["answer"]
                sources = result["sources"]
                standalone_q = result.get("standalone_question")
                pipeline_info = result.get("pipeline_info", {})

                # Typing effect
                words = answer.split()
                for i, word in enumerate(words):
                    full_response += word + " "
                    if i % 3 == 0:
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)

                message_placeholder.markdown(full_response)

                if standalone_q:
                    st.caption(f'🔍 Interpreted as: "{standalone_q}"')

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"{i}. `{src}`")

                if pipeline_info:
                    with st.expander("🔬 Pipeline info", expanded=False):
                        cols = st.columns(3)
                        cols[0].metric("Retrieved", pipeline_info.get("total_retrieved", 0))
                        cols[1].metric("Used", pipeline_info.get("final_docs", 0))
                        cols[2].metric("Context", "✅" if pipeline_info.get("contextualized") else "—")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                full_response = "An error occurred. Please try again."
                message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat(final_notebook_name, st.session_state.messages)
