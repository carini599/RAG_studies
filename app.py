import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_documents
from rag.retriever import Retriever
from rag.generator import generate_answer

# ---------- Setup ----------
load_dotenv()

st.set_page_config(page_title="RAG Studies", layout="wide")

st.title("RAG Studies by Carina Breu📚🤖")
st.write("Ask questions about your documents.")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 5)
max_distance = st.sidebar.slider(
    "Max Distance (Relevance Threshold)",
    0.1, 2.0, 1.2, 0.1
)

# ---------- Load documents ----------
@st.cache_resource
def setup_retriever():
    chunks = load_documents()
    return Retriever(chunks), chunks

with st.spinner("Loading documents and building index..."):
    retriever, all_chunks = setup_retriever()

st.sidebar.success(f"Loaded {len(all_chunks)} chunks")

# ---------- User Input ----------
question = st.text_input(
    "❓ Ask a question",
    placeholder="e.g. What is RAG and why is it useful?"
)


# ---------- Run RAG ----------
if question:
    with st.spinner("Searching and generating answer..."):
        retrieved_chunks = retriever.retrieve(
            question,
            k=top_k,
            max_distance=max_distance
        )

        answer = generate_answer(question, retrieved_chunks)

    # ---------- Answer ----------
    st.subheader("🤖 Answer")
    st.write(answer)

    # ---------- Sources ----------
    st.subheader("📚 Sources")

    if not retrieved_chunks:
        st.warning("No relevant sources found.")
    else:
        for i, c in enumerate(retrieved_chunks, start=1):
            with st.expander(f"[{i}] {c['source']} (page {c['page']})"):
                st.write(c["text"])
                st.caption(f"Distance: {c['distance']:.3f}")

