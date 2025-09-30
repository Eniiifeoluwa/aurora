import sys, os, time
import streamlit as st
from aurora.rag_pipeline import RAGPipeline
from aurora.utils.loaders import load_text_file, load_pdf, load_image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
sys.path.append(os.path.dirname(__file__))

MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY!")
banner = Image.open("assets/aurora.png")

st.set_page_config(page_title="Aurora — Intelligent AI Assistant", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

PERSIST_DIR = "./chroma_db"

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(persist_directory=PERSIST_DIR)

pipeline = st.session_state.pipeline

if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%); color: #e0e0e0 !important; padding: 2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d0d 0%, #1a1a2e 100%); border-right: 1px solid rgba(255,0,128,0.3); }
    h1 { font-size: 3.2rem !important; font-weight: 800 !important; background: linear-gradient(135deg,#3b82f6 0%,#a855f7 50%,#ef4444 100%);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; text-align: center; }
    .stTextInput > div > div > input { background: rgba(30,30,60,0.8); border: 2px solid rgba(99,102,241,0.6); border-radius: 12px; padding: 14px 18px; color: #f8fafc; }
    .stButton > button { background: linear-gradient(135deg,#3b82f6 0%,#9333ea 50%,#ef4444 100%); color:white; border:none; border-radius:12px; padding:14px 28px; font-weight:600; transition: all 0.3s; }
    .stButton > button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 8px 24px rgba(147,51,234,0.7); }
    .streamlit-expanderHeader { background: rgba(50,0,70,0.5); color: #a855f7; border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Aurora</h1>", unsafe_allow_html=True)
st.image(banner, use_container_width=True)
st.markdown("<p style='text-align:center; font-size:1.2rem; color:white;'>An Intelligent AI Assistant built by Akinola Afolabi</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📁 Document Management")
    uploaded = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['txt', 'pdf', 'png', 'jpg', 'jpeg'])
    col1, col2 = st.columns(2)

    with col1:
        @st.cache_data(show_spinner=False)
        def process_uploaded_file(f):
            name, b = f.name, f.read()
            if name.endswith(".txt"):
                return load_text_file(b, name)
            elif name.endswith(".pdf"):
                return load_pdf(b, name)
            elif name.lower().endswith((".png", ".jpg", ".jpeg")):
                return load_image(b, name)
            else:
                return []

        if st.button("📥 Index Files", use_container_width=True):
            if uploaded:
                all_docs = []
                for f in uploaded:
                    docs = process_uploaded_file(f)
                    all_docs.extend(docs)
                try:
                    pipeline.index_documents(all_docs, persist=True)
                    st.session_state.docs_indexed = True
                    st.success(f"✅ Indexed {len(all_docs)} documents!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Indexing failed: {e}")
            else:
                st.warning("⚠️ Please upload files first")
    with col2:
        if st.button("💾 Load Index", use_container_width=True):
            try:
                with st.spinner("Loading index..."):
                    pipeline.load_index()
                st.session_state.docs_indexed = True
                st.success("✅ Index loaded!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {e}")

    st.markdown("---")
    if st.session_state.docs_indexed:
        st.success("🟢 Documents Ready")
    else:
        st.warning("🟡 No Documents Indexed")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    k = st.slider("Retrieved chunks (k)", 1, 10, 4)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.0, step=0.1)
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("History cleared!")
        time.sleep(0.5)
        st.rerun()

if not st.session_state.docs_indexed:
    st.info("ℹ️ Upload and index documents in the sidebar to start asking questions.")
else:
    st.markdown("### 💬 Ask Aurora Anything")
    query = st.text_input("Ask a question", placeholder="What would you like to know?", label_visibility="collapsed")
    run_button = st.button("🚀 Ask", use_container_width=True)

    if run_button and query.strip():
        @st.cache_data(show_spinner=False)
        def cached_query(_pipeline, query_text, k):
            return _pipeline.query(query_text, k)

        try:
            context, docs = cached_query(pipeline, query, k)
            with st.expander(f"📚 Retrieved Context ({len(docs)} chunks)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i} — {d.metadata.get('source','unknown')}**")
                    st.text(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
            result = llm([SystemMessage(content="You are Aurora, a brilliant AI assistant built by Olamidipupo, Akinola, Afolabi (You can use any of these names), a reputable AI Engineer. Use context. If unknown, say so."), HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")])
            st.markdown("### ✨ Answer")
            st.markdown(f"<div style='padding:1.5rem;background:rgba(99,102,241,0.1);border-radius:12px'>{result.content}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append({"question": query, "answer": result.content, "sources": len(docs)})
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Conversations")
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"Q: {chat['question'][:60]}..."):
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"📊 Sources used: {chat['sources']}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:white;'>Powered by <b>Groq</b> • <b>LangChain</b> • <b>Chroma</b></p>", unsafe_allow_html=True)
