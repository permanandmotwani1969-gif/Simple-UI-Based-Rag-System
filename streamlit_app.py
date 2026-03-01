import streamlit as st
import faiss
import numpy as np
import re
import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="RAG AI Assistant")

st.title("📄 RAG AI Chat Assistant")

# 🔵 Load PDF once
@st.cache_resource
def load_rag_system():

    reader = PdfReader("Rag.pdf")
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    def sentence_chunk_text(text, max_chars=500):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    chunks = sentence_chunk_text(text)

    embedder = SentenceTransformer("all-mpnet-base-v2")
    chunk_embeddings = embedder.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings)
    faiss.normalize_L2(chunk_embeddings)

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    return chunks, embedder, index

chunks, embedder, index = load_rag_system()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

query = st.text_input("Ask your question")

if query:

    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Using ONLY the context below, answer clearly in 3-4 sentences.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    confidence = round(float(distances[0][0]), 2)

    st.subheader("AI Answer")
    st.write(answer)

    st.caption(f"Confidence: {confidence}")
    st.caption(f"Tokens used: {response.usage.total_tokens}")