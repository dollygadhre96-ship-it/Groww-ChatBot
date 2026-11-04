"""Streamlit chat UI for the Groww Mutual Fund FAQ Assistant."""
from __future__ import annotations

import html
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import streamlit as st
import yaml

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation import GroqAnswerGenerator
from src.chunking.chunker import chunk_documents
from src.guardrails import check_input, validate_answer
from src.ingestion.loader import load_documents
from src.retrieval import Retriever
from src.vector_store.store import ChromaStore

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Groww MF FAQ Assistant", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.stApp { background: #f8f9fc; color: #292747; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 5% 0%, rgba(128, 112, 224, .11), transparent 34rem),
    radial-gradient(ellipse at 98% 12%, rgba(75, 193, 167, .12), transparent 31rem),
    linear-gradient(180deg, #f7f4fc 0%, #f1f8f5 52%, #f7f4fa 100%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px; padding: 1.55rem 1.25rem 2.6rem; }
.app-header { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 1.05rem; padding: 1.2rem 1.35rem; margin: .15rem 0 1.2rem; border: 1px solid #dfd1e8; border-radius: 1.4rem; background: linear-gradient(112deg, #f9e9ca 0%, #eee2f8 48%, #d9eee4 100%); box-shadow: 0 12px 30px rgba(100,82,115,.11); }
.brand-mark { display: grid; place-items: center; width: 4rem; height: 4rem; border: 3px solid #fff8ed; border-radius: 1.3rem; background: linear-gradient(145deg, #9b6848 0%, #75482f 100%); color: #ffffff; font-size: 1.9rem; box-shadow: 0 0 0 4px rgba(224,194,154,.42), 0 9px 22px rgba(112,74,48,.22); }
.brand-kicker { display: inline-flex; margin: 0 0 .4rem; padding: .27rem .62rem; border: 1px solid #ded9f6; border-radius: 999px; background: #efedff; color: #5f51bb; font-size: .68rem; font-weight: 780; letter-spacing: .12em; text-transform: uppercase; }
.app-title { margin: 0; color: #292747; font-size: clamp(1.4rem, 3vw, 2rem); font-weight: 760; letter-spacing: -.04em; line-height: 1.12; }
.app-subtitle { margin: .42rem 0 0; color: #686680; font-size: .94rem; }
.header-badge { display: inline-flex; align-items: center; gap: .48rem; padding: .55rem .78rem; border: 1px solid #e2e4ed; border-radius: 999px; background: rgba(255,255,255,.86); color: #54536e; font-size: .75rem; font-weight: 650; white-space: nowrap; }
.badge-dot { width: .48rem; height: .48rem; border-radius: 50%; background: #36bd91; box-shadow: 0 0 0 4px rgba(54,189,145,.13); }
.welcome-card { margin: .2rem 0 1.3rem; padding: 1.45rem 1.55rem; border: 1px solid #ded7ec; border-left: 4px solid #e89b78; border-radius: 1.25rem; background: linear-gradient(118deg, #fffdfa 0%, #f3effa 72%, #e8f5f0 100%); box-shadow: 0 10px 25px rgba(71,65,130,.08); }
.welcome-eyebrow { display: inline-flex; margin: 0 0 .5rem; padding: .28rem .58rem; border-radius: 999px; background: #efedff; color: #6254bd; font-size: .7rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
.welcome-title { margin: 0 0 .45rem; color: #292747; font-size: 1.48rem; font-weight: 750; letter-spacing: -.035em; }
.welcome-copy { max-width: 46rem; margin: 0; color: #66647d; line-height: 1.62; }
.welcome-tags { display: flex; flex-wrap: wrap; gap: .48rem; margin-top: 1rem; }
.welcome-tag { padding: .34rem .62rem; border: 1px solid #e6e3f2; border-radius: 999px; background: rgba(255,255,255,.88); color: #56536f; font-size: .72rem; font-weight: 650; }
.welcome-tag:nth-child(1) { border-color: #cceee4; background: #edfaf6; color: #287c69; }
.welcome-tag:nth-child(2) { border-color: #e1dcf8; background: #f3f1ff; color: #675bb0; }
.welcome-tag:nth-child(3) { border-color: #f6e1d2; background: #fff7f0; color: #a66b47; }
.section-label { margin: 1.15rem 0 .65rem; color: #6157aa; font-size: .79rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
.stButton > button { min-height: 4.9rem; padding: .9rem 1rem !important; border: 1px solid #e3e0f1 !important; border-radius: 1.05rem !important; background: linear-gradient(145deg, #ffffff 0%, #faf9ff 100%) !important; color: #302e4d !important; box-shadow: 0 6px 17px rgba(66,61,120,.06); font-weight: 650 !important; text-align: left !important; white-space: normal !important; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease; }
.stButton > button:hover { transform: translateY(-2px); border-color: #b9afea !important; background: linear-gradient(145deg, #ffffff 0%, #f5f2ff 100%) !important; box-shadow: 0 10px 21px rgba(76,69,139,.1); }
.stButton > button:focus { border-color: #9588db !important; box-shadow: 0 0 0 3px rgba(149,136,219,.16) !important; }
[data-testid="stChatMessage"] { margin: .8rem 0; padding: 1.08rem 1.2rem; border: 1px solid #e8e6f0; border-left: 4px solid #b9afea; border-radius: 1.15rem; background: rgba(255,255,255,.96); box-shadow: 0 7px 18px rgba(54,50,98,.055); }
[data-testid="stChatInput"] { border: 1px solid #d9cbbf; border-radius: 1.15rem; background: #fffdf9; box-shadow: 0 8px 20px rgba(104,77,53,.1); }
[data-testid="stChatInput"]:focus-within { border-color: #a97856; box-shadow: 0 0 0 3px rgba(169,120,86,.16), 0 8px 20px rgba(104,77,53,.1); }
[data-testid="stChatInput"] textarea { color: #302e4d; }
[data-testid="stChatInput"] button { border: 1px solid #805136; border-radius: .8rem; background: linear-gradient(135deg, #9b6848, #75482f); color: #ffffff; }
.source-heading { margin: 1rem 0 .55rem; color: #675dae; font-size: .72rem; font-weight: 850; letter-spacing: .12em; text-transform: uppercase; }
a.source-card { display: flex; align-items: center; gap: .75rem; margin: .48rem 0; padding: .8rem .92rem; border: 1px solid #e5e1f1; border-radius: 1rem; background: linear-gradient(105deg, #ffffff 0%, #faf9ff 74%, #f3fbf8 100%); color: #302e4d; text-decoration: none !important; font-size: .86rem; line-height: 1.45; box-shadow: 0 4px 13px rgba(66,61,120,.045); transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease; }
a.source-card:hover { transform: translateY(-1px); border-color: #b9afea; box-shadow: 0 8px 18px rgba(76,69,139,.09); }
.source-index { display: grid; place-items: center; width: 1.85rem; height: 1.85rem; flex: 0 0 1.85rem; border-radius: .62rem; background: linear-gradient(145deg, #eeebff, #e6f8f2); color: #6259a8; font-size: .7rem; font-weight: 850; }
.source-title { flex: 1; }
.source-arrow { color: #3aab91; font-size: 1rem; font-weight: 800; }
.facts-footer { margin-top: 1.8rem; padding: .8rem 1rem; border: 1px solid #e9e6f1; border-radius: 999px; background: rgba(255,255,255,.82); color: #6a6880; font-size: .74rem; text-align: center; }
@media (max-width: 680px) {
  .block-container { padding: 1rem .8rem 2rem; }
  .app-header { grid-template-columns: auto 1fr; gap: .75rem; padding: .9rem; }
  .brand-mark { width: 2.85rem; height: 2.85rem; border-radius: .95rem; }
  .header-badge { display: none; }
  .app-subtitle { font-size: .83rem; }
  .welcome-card { padding: 1.05rem; }
  .stButton > button { min-height: 3.6rem; }
  .facts-footer { border-radius: .8rem; }
}
@media (prefers-reduced-motion: reduce) {
  .stButton > button, a.source-card { transition: none; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))


@st.cache_resource
def get_answer_generator(model_name: str):
    try:
        hosted_api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        hosted_api_key = ""
    return GroqAnswerGenerator(model_name, api_key=hosted_api_key)


@st.cache_resource
def get_embedder(model_name: str):
    # Load the heavyweight model only when a question needs retrieval.
    from src.embeddings.service import EmbeddingService
    return EmbeddingService(model_name)


@st.cache_resource
def get_retriever(
    model_name: str,
    persist_directory: str,
    collection_name: str,
    top_k: int,
    relevance_threshold: float,
    chunk_size: int,
    chunk_overlap: int,
):
    # Community Cloud runs on Linux and its local disk is temporary. Keep its
    # rebuildable Chroma index under /tmp; retain the configured path locally.
    if os.name == "nt":
        persist_path = ROOT / persist_directory
    else:
        persist_path = Path(tempfile.gettempdir()) / "groww-chatbot-chroma"

    store = ChromaStore(persist_path, collection_name, model_name)
    documents = None
    marker_path = None

    if os.name != "nt":
        # On a redeploy, /tmp may survive even when its source snapshot changes.
        # A small fingerprint lets the cloud rebuild only when the seed docs change.
        documents = load_documents(ROOT / "data" / "processed")
        fingerprint_input = "\n".join(
            f"{doc.metadata['source_id']}:{doc.metadata['content_hash']}"
            for doc in sorted(documents, key=lambda item: item.metadata["source_id"])
        )
        source_fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
        marker_path = persist_path.parent / f"{collection_name}.source.sha256"
        saved_fingerprint = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
        if saved_fingerprint != source_fingerprint and store.count() > 0:
            store.reset()

    embedder = get_embedder(model_name)
    if store.count() == 0:
        if documents is None:
            documents = load_documents(ROOT / "data" / "processed")
        if not documents:
            raise RuntimeError("No bundled Groww source documents were found in data/processed.")
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise RuntimeError("The bundled Groww source documents produced no searchable chunks.")
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        store.upsert_chunks(chunks, vectors)
        logger.info("Built searchable index with %s chunks from %s sources", len(chunks), len(documents))
    if marker_path is not None:
        marker_path.write_text(source_fingerprint, encoding="utf-8")

    return Retriever(
        store,
        embedder,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )


def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    st.markdown('<div class="source-heading">Sources</div>', unsafe_allow_html=True)
    for index, source in enumerate(sources, 1):
        title = html.escape(str(source.get("title") or "Groww source"), quote=True)
        url = html.escape(str(source.get("url") or ""), quote=True)
        if not url:
            continue
        st.markdown(
            f'<a class="source-card" href="{url}" target="_blank" rel="noopener noreferrer">'
            f'<span class="source-index">{index:02}</span>'
            f'<span class="source-title">{title}</span>'
            '<span class="source-arrow">↗</span></a>',
            unsafe_allow_html=True,
        )


cfg = load_config()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

header, controls = st.columns([6, 1], vertical_alignment="center")
with header:
    st.markdown(
        '<div class="app-header"><div class="brand-mark">🌱</div><div>'
        '<p class="brand-kicker">Groww · Mutual Funds</p>'
        '<h1 class="app-title">HDFC Mutual Fund Assistant</h1>'
        '<p class="app-subtitle">Simple, source-backed answers to your scheme questions.</p>'
        '</div><div class="header-badge"><span class="badge-dot"></span>5 Groww source pages</div></div>',
        unsafe_allow_html=True,
    )
with controls:
    if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome-card"><p class="welcome-eyebrow">Your HDFC fund guide</p>'
        '<p class="welcome-title">What would you like to know?</p>'
        '<p class="welcome-copy">Ask about scheme details like expense ratios, exit loads, SIPs, benchmarks, or ELSS lock-in periods.</p>'
        '<div class="welcome-tags"><span class="welcome-tag">Scheme facts</span><span class="welcome-tag">Source-linked</span><span class="welcome-tag">No investment advice</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Start with a popular question</div>', unsafe_allow_html=True)
    examples = [
        "📈  What is the exit load of HDFC Large Cap Fund?",
        "🔒  What is the lock-in period for HDFC ELSS?",
        "◈  What is the benchmark of HDFC Mid Cap Fund?",
    ]
    cols = st.columns(3)
    for index, (col, question) in enumerate(zip(cols, examples)):
        with col:
            if st.button(question, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_query = question

for message in st.session_state.messages:
    avatar = "🌱" if message["role"] == "assistant" else "🧑"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

query = st.chat_input("Ask a factual HDFC Mutual Fund question…")
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    check = check_input(query)
    trace = {
        "query": query,
        "input_check": {"allowed": check.allowed, "category": check.category, "message": check.message},
    }
    answer = check.message
    sources: list[dict[str, str]] = []

    if not check.allowed:
        trace["retrieval"] = {"skipped": True}
        trace["generation"] = {"status": "skipped", "reason": check.category}
        trace["validation"] = {"status": "not_applicable", "reason": check.category}
    else:
        try:
            with st.spinner("Preparing the source index and finding evidence…"):
                e_cfg = cfg.get("embeddings", {})
                v_cfg = cfg.get("vector_store", {})
                r_cfg = cfg.get("retrieval", {})
                chunk_cfg = cfg.get("chunking", {})
                model_name = e_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
                retriever = get_retriever(
                    model_name=model_name,
                    persist_directory=v_cfg.get("persist_directory", "data/chroma"),
                    collection_name=v_cfg.get("collection_name", "hdfc_mf_faq"),
                    top_k=int(r_cfg.get("top_k", 5)),
                    relevance_threshold=float(r_cfg.get("relevance_threshold", .45)),
                    chunk_size=int(chunk_cfg.get("chunk_size", 800)),
                    chunk_overlap=int(chunk_cfg.get("chunk_overlap", 120)),
                )
                result = retriever.retrieve(check.safe_query)

            sources = result.sources
            trace["retrieval"] = {
                "top_k": retriever.top_k,
                "threshold": result.threshold,
                "no_evidence": result.no_evidence,
                "chunks": [
                    {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance, "relevance": c.relevance}
                    for c in result.chunks
                ],
                "context": result.context,
                "sources": result.sources,
            }

            if result.no_evidence:
                answer = "I couldn't find this in the available Groww sources. Try asking about a specific HDFC Mutual Fund scheme detail."
                trace["generation"] = {"status": "skipped", "reason": "no_evidence"}
                trace["validation"] = {"status": "not_applicable", "reason": "no_evidence"}
            else:
                llm_cfg = cfg.get("llm", {})
                model_name = llm_cfg.get("model", "qwen/qwen3.8-27b")
                max_sentences = int(llm_cfg.get("max_answer_sentences", 3))
                try:
                    with st.spinner("Writing an answer from the retrieved evidence…"):
                        answer = get_answer_generator(model_name).generate(
                            check.safe_query, result.context, max_sentences=max_sentences
                        )
                    valid, reason = validate_answer(answer, result.sources, max_sentences=max_sentences)
                    trace["generation"] = {"status": "complete", "model": model_name}
                    trace["validation"] = {"status": "passed" if valid else "rejected", "reason": reason}
                    if not valid:
                        answer = "I couldn't produce a grounded answer that passed the source and length checks. Try rephrasing your question."
                except Exception as exc:
                    logger.exception("Groq answer generation failed for model %s", model_name)
                    answer = "I couldn't generate a grounded answer. Check GROQ_API_KEY in your Streamlit app's Settings → Secrets, then review the app logs if the problem continues."
                    trace["generation"] = {
                        "status": "failed",
                        "error": type(exc).__name__,
                        "cause": type(exc.__cause__).__name__ if exc.__cause__ else None,
                    }
                    trace["validation"] = {"status": "not_run", "reason": "generation_failed"}
        except Exception as exc:
            logger.exception("Retrieval failed while answering a user query")
            answer = "The retrieval service isn't ready. Check that the embedding model is installed and the Chroma collection has been populated."
            trace["error"] = type(exc).__name__
            trace["generation"] = {"status": "not_attempted", "reason": "retrieval_failed"}
            trace["validation"] = {"status": "not_run", "reason": "retrieval_failed"}

    trace["answer"] = answer
    trace.setdefault("validation", {"status": "not_applicable", "reason": "generation was skipped"})
    st.session_state.last_trace = trace
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    with st.chat_message("assistant", avatar="🌱"):
        st.markdown(answer)
        render_sources(sources)

st.markdown(
    '<div class="facts-footer">Facts-only information from indexed Groww pages · No investment advice</div>',
    unsafe_allow_html=True,
)
