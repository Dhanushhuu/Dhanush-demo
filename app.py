
import streamlit as st, time, sys, os
from datetime import datetime
sys.path.insert(0, "/tmp/cvip_app")

st.set_page_config(page_title="CVIP RAG", page_icon="🤖", layout="wide")
st.markdown("""<style>
.answer-box{background:#f5f7fa;padding:1.5rem;border-radius:12px;
border-left:5px solid #1E88E5;margin:1rem 0;line-height:1.7;}
.citation-box{background:#fff9e6;padding:.75rem;border-radius:8px;
border-left:3px solid #ffc107;margin:.3rem 0;font-size:.9rem;}
</style>""", unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "session_id"   not in st.session_state: st.session_state.session_id=f"ui_{int(time.time())}"
if "initialized"  not in st.session_state: st.session_state.initialized=False
if "rag"          not in st.session_state: st.session_state.rag=None

@st.cache_resource
def load_rag():
    os.environ.setdefault("DATABRICKS_HOST","https://dbc-9d1ce33e-6acf.cloud.databricks.com")
    with open("/tmp/cvip_app/rag_components.py") as f:
        source = f.read()
    namespace = {"__name__": "__main__"}
    exec(compile(source, "rag_components.py", "exec"), namespace)
    FinalProductionRAG = namespace["FinalProductionRAG"]
    return FinalProductionRAG(
        enable_reranking=False, enable_persistence=True,
        flush_every_n=3, flush_every_seconds=30
    )

with st.sidebar:
    st.title("⚙️ Controls")
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Loading… (2-3 min first time)"):
                try:
                    st.session_state.rag = load_rag()
                    st.session_state.initialized = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
    else:
        st.success("✅ System Online")
    st.markdown("---")
    show_sources = st.checkbox("📚 Show Sources", value=True)
    show_debug   = st.checkbox("🔍 Debug Info",   value=False)
    st.markdown("---")
    if st.button("🔄 New Chat", use_container_width=True):
        if st.session_state.rag:
            try: st.session_state.rag.cleanup(st.session_state.session_id)
            except: pass
        st.session_state.chat_history = []
        st.session_state.session_id   = f"ui_{int(time.time())}"
        st.rerun()
    if st.session_state.initialized and st.session_state.rag:
        st.markdown("---")
        try:
            s = st.session_state.rag.get_system_stats()
            st.metric("Sessions", s.get("active_sessions",0))
            st.metric("Uptime",   s.get("system_uptime","-"))
        except: pass

st.markdown("# 🤖 CVIP RAG System")
st.markdown("*Computer Vision & Image Processing Expert*")
st.markdown("---")

if not st.session_state.initialized:
    st.info("👈 Click **Initialize System** in the sidebar to begin.")
    st.stop()

EXAMPLES=["What is edge detection?","How does Sobel work?",
          "Explain CNNs","What are vision transformers?","Compare CNN vs traditional CV"]

if not st.session_state.chat_history:
    st.markdown("### 💡 Example Questions")
    cols=st.columns(len(EXAMPLES))
    for col,ex in zip(cols,EXAMPLES):
        with col:
            if st.button(ex, use_container_width=True):
                st.session_state.pending=ex
                st.rerun()
    st.markdown("---")

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["query"])
        st.caption(entry["time"])
    with st.chat_message("assistant"):
        r=entry["response"]
        st.markdown(f'<div class="answer-box">{r["answer"]}</div>', unsafe_allow_html=True)
        if show_sources and r.get("citations"):
            with st.expander("📚 Sources"):
                for i,c in enumerate(r["citations"],1):
                    st.markdown(f'<div class="citation-box">{i}. {c}</div>', unsafe_allow_html=True)
        if r["support_level"] not in ["memory_recall","out_of_domain","error"]:
            c1,c2,c3=st.columns(3)
            c1.metric("Support",    r["support_level"])
            c2.metric("Confidence", f'{r["confidence"]:.1%}')
            c3.metric("Latency",    f'{r["latency_ms"]}ms')
        if show_debug:
            with st.expander("🐛 Debug"):
                st.json({"intent":r["query_classification"]["intent"],
                         "chunks":len(r.get("retrieved_chunks",[])),
                         "grounding":r.get("grounding_scores",{})})

query=None
if hasattr(st.session_state,"pending"):
    query=st.session_state.pending
    del st.session_state.pending
else:
    query=st.chat_input("Ask about Computer Vision or Image Processing…")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r=st.session_state.rag.ask(
                    query=query,
                    session_id=st.session_state.session_id,
                    use_reranking=False
                )
                st.session_state.chat_history.append({
                    "query":query,"response":r,
                    "time":datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

st.markdown("---")
st.caption("🤖 CVIP RAG | Databricks Vector Search + LLaMA 3.3 70B")
