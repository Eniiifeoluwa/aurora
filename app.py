import sys, os
sys.path.append(os.path.dirname(__file__))
import streamlit as st
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY! Please add it in Streamlit secrets or environment.")

from aurora.rag_pipeline import RAGPipeline
from aurora.utils.loaders import load_text_file, load_pdf, load_image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import time
# Page config with custom theme
st.set_page_config(
    page_title="Aurora — Intelligent RAG Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #e0e0e0 !important;
        padding: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d0d 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 0, 128, 0.3);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #9d4edd !important;
    }

    /* Title styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #3b82f6 0%, #a855f7 50%, #ef4444 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 1rem !important;
        text-align: center !important;
        letter-spacing: -0.02em !important;
    }

    /* Card containers */
    .stContainer, div[data-testid="stVerticalBlock"] > div {
        background: rgba(20, 20, 35, 0.9) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.4) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        margin-bottom: 2rem !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(30, 30, 60, 0.8) !important;
        border: 2px solid rgba(99, 102, 241, 0.6) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 16px !important;
        color: #f8fafc !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.5) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #9333ea 50%, #ef4444 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(147, 51, 234, 0.7) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(50, 0, 70, 0.5) !important;
        color: #a855f7 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #a855f7, #ef4444);
        border-radius: 4px;
    }

    /* Retrieved context blocks */
    .retrieved-context {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #f1f5f9 !important;
    }

    /* Answers */
    .answer-box {
        padding: 2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(147, 51, 234, 0.15), rgba(239, 68, 68, 0.15));
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #f8fafc !important;
        line-height: 1.8;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>✨ Aurora</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <p style='font-size: 1.3rem; color: white; font-weight: 500; text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>
        Intelligent RAG Assistant powered by Groq & LangChain
    </p>
</div>
""", unsafe_allow_html=True)

# Instantiate pipeline
PERSIST_DIR = "./faiss_index"
pipeline = RAGPipeline(persist_directory=PERSIST_DIR)

if "docs_indexed" not in st.session_state:
    st.session_state["docs_indexed"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
        help="Supported formats: TXT, PDF, PNG, JPG, JPEG"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Index Files", use_container_width=True):
            if uploaded:
                with st.spinner("Processing documents..."):
                    all_docs = []
                    progress_bar = st.progress(0)
                    
                    for idx, f in enumerate(uploaded):
                        name = f.name
                        b = f.read()
                        
                        if name.lower().endswith(".txt"):
                            docs = load_text_file(b, name)
                        elif name.lower().endswith(".pdf"):
                            docs = load_pdf(b, name)
                        elif name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                            docs = load_image(b, name)
                        else:
                            st.warning(f"⚠️ Unsupported: {name}")
                            continue
                        
                        all_docs.extend(docs)
                        progress_bar.progress((idx + 1) / len(uploaded))
                    
                    if all_docs:
                        pipeline.index_documents(all_docs, persist=True)
                        st.session_state["docs_indexed"] = True
                        st.success(f"✅ Indexed {len(all_docs)} documents!")
                        time.sleep(1)
                        st.rerun()
            else:
                st.warning("⚠️ Please upload files first")
    
    with col2:
        if st.button("💾 Load Index", use_container_width=True):
            try:
                with st.spinner("Loading index..."):
                    pipeline.load_index()
                    st.session_state["docs_indexed"] = True
                    st.success("✅ Index loaded!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {str(e)[:50]}")
    
    st.markdown("---")
    
    # Status indicator
    if st.session_state["docs_indexed"]:
        st.markdown("""
        <div style='padding: 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 12px; border-left: 4px solid #22c55e;'>
            <p style='margin: 0; color: #166534; font-weight: 600;'>🟢 Documents Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding: 1rem; background: rgba(251, 146, 60, 0.1); border-radius: 12px; border-left: 4px solid #fb923c;'>
            <p style='margin: 0; color: #9a3412; font-weight: 600;'>🟡 No Documents Indexed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    k = st.slider(
        "Retrieved chunks (k)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of document chunks to retrieve"
    )
    
    temperature = st.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.success("History cleared!")
        time.sleep(0.5)
        st.rerun()

# Main content area
if not st.session_state["docs_indexed"]:
    st.info("ℹ️ **Get Started:** Upload and index documents in the sidebar to begin asking questions!")
else:
    # Query input section
    st.markdown("### 💬 Ask Aurora Anything")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
    with col2:
        run_button = st.button("🚀 Ask", use_container_width=True, type="primary")
    
    # Process query
    if run_button and query.strip():
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                context, docs = pipeline.query(query, k=k)
                
                # Show retrieved context in expander
                with st.expander(f"📚 Retrieved Context ({len(docs)} chunks)", expanded=False):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"""
                        <div style='padding: 1rem; background: rgba(99, 102, 241, 0.05); border-radius: 12px; margin-bottom: 1rem; border-left: 3px solid #667eea;'>
                            <p style='margin: 0; font-weight: 600; color: #667eea;'>
                                Chunk {i} — {d.metadata.get('source', 'unknown')}
                            </p>
                            <p style='margin-top: 0.5rem; color: #475569; line-height: 1.6;'>
                                {d.page_content[:500]}{"..." if len(d.page_content) > 500 else ""}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Generate answer
                with st.spinner("🤖 Generating answer..."):
                    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=temperature)
                    
                    system_msg = SystemMessage(
                        content="You are Aurora, a brilliant and helpful AI assistant. Use the provided context to answer questions accurately. "
                                "If the answer isn't in the context, politely say you don't know. "
                                "Be concise, clear, and cite sources when relevant."
                    )
                    
                    human_msg = HumanMessage(
                        content=f"""Context:
{context}

Question: {query}

Provide a clear, well-structured answer. Use markdown formatting for readability.
"""
                    )
                    
                    result = llm([system_msg, human_msg])
                    
                    # Display answer
                    st.markdown("### ✨ Answer")
                    st.markdown(f"""
                    <div style='padding: 2rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%); 
                                border-radius: 16px; border: 2px solid rgba(99, 102, 241, 0.2);'>
                        <p style='color: #1e293b; line-height: 1.8; font-size: 1.05rem;'>
                            {result.content}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state["chat_history"].append({
                        "question": query,
                        "answer": result.content,
                        "sources": len(docs)
                    })
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Chat history
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.markdown("### 📜 Recent Conversations")
        
        for idx, chat in enumerate(reversed(st.session_state["chat_history"][-5:]), 1):
            with st.expander(f"Q{idx}: {chat['question'][:60]}...", expanded=False):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"📊 Sources used: {chat['sources']}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <p style='color: white; font-size: 0.9rem; text-shadow: 0 1px 2px rgba(0,0,0,0.3);'>
        Powered by <strong>Groq</strong> • <strong>LangChain</strong> • <strong>FAISS</strong>
    </p>
</div>
""", unsafe_allow_html=True)