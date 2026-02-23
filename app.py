
import streamlit as st
import requests
import time
import os
from datetime import datetime

st.set_page_config(page_title="CVIP RAG", page_icon="🤖", layout="wide")
st.markdown("""<style>
.answer-box{background:#f5f7fa;padding:1.5rem;border-radius:12px;
border-left:5px solid #1E88E5;margin:1rem 0;line-height:1.7;}
.citation-box{background:#fff9e6;padding:.75rem;border-radius:8px;
border-left:3px solid #ffc107;margin:.3rem 0;font-size:.9rem;}
</style>""", unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history=[]
if "session_id"   not in st.session_state: st.session_state.session_id=f"ui_{int(time.time())}"

DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST",  "https://dbc-9d1ce33e-6acf.cloud.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
LLM_ENDPOINT     = "databricks-meta-llama-3-3-70b-instruct"
VECTOR_INDEX     = "workspace.default.cvip_chunks_vs_index"
VECTOR_ENDPOINT  = "cvip_endpoint"

def query_llm(query, context):
    import mlflow.deployments
    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": (
                    "You are an expert in Computer Vision and Image Processing. "
                    "Answer ONLY using the provided context. "
                    "Cite sources using [Source: name] format."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 800,
            "temperature": 0.1,
            "stream": False
        }
    )
    return response["choices"][0]["message"]["content"]

def query_vector_search(query):
    from databricks.vector_search.client import VectorSearchClient
    vsc   = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=VECTOR_ENDPOINT,
        index_name=VECTOR_INDEX
    )
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_id","content","citation_label","page_number"],
        num_results=5
    )
    chunks = []
    for row in results.get("result",{}).get("data_array",[]):
        chunks.append({
            "content":        row[1] if len(row)>1 else "",
            "citation_label": row[2] if len(row)>2 else "Unknown",
            "page_number":    row[3] if len(row)>3 else ""
        })
    return chunks

def ask(query):
    chunks  = query_vector_search(query)
    if not chunks:
        return {"answer": "No relevant information found.", "citations": [], "latency_ms": 0}
    context = "\n\n".join([
        f"[Source: {c['citation_label']}]\n{c['content'][:500]}"
        for c in chunks
    ])
    start   = time.time()
    answer  = query_llm(query, context)
    latency = int((time.time()-start)*1000)
    import re
    citations = re.findall(r"\[Source:([^\]]+)\]", answer)
    return {"answer": answer, "citations": citations, "latency_ms": latency, "chunks": len(chunks)}

# Sidebar
with st.sidebar:
    st.title("⚙️ CVIP RAG")
    st.success("✅ System Ready")
    st.markdown("---")
    show_sources = st.checkbox("📚 Show Sources", value=True)
    st.markdown("---")
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.chat_history=[]
        st.session_state.session_id=f"ui_{int(time.time())}"
        st.rerun()
    st.markdown("---")
    st.markdown("### 💡 Examples")
    EXAMPLES=["What is edge detection?","How does Sobel work?",
              "Explain CNNs","What are vision transformers?"]
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending=ex
            st.rerun()

st.markdown("# 🤖 CVIP RAG System")
st.markdown("*Computer Vision & Image Processing Expert*")
st.markdown("---")

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["query"])
        st.caption(entry["time"])
    with st.chat_message("assistant"):
        r=entry["response"]
        st.markdown(f'<div class="answer-box">{r["answer"]}</div>',unsafe_allow_html=True)
        if show_sources and r.get("citations"):
            with st.expander("📚 Sources"):
                for i,c in enumerate(r["citations"],1):
                    st.markdown(f'<div class="citation-box">{i}. {c}</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        c1.metric("Chunks Retrieved", r.get("chunks",0))
        c2.metric("Latency", f'{r["latency_ms"]}ms')

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
                r=ask(query)
                st.session_state.chat_history.append({
                    "query":query,"response":r,
                    "time":datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

st.markdown("---")
st.caption("🤖 CVIP RAG | Databricks Vector Search + LLaMA 3.3 70B")
