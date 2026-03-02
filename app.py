import streamlit as st
import numpy as np
import re
import os
import tempfile

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Try different FAISS import methods
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS as LangchainFAISS
        FAISS_AVAILABLE = False
        st.warning("Using LangChain FAISS wrapper")
    except:
        FAISS_AVAILABLE = False
        st.error("FAISS not available - using numpy fallback")

st.set_page_config(page_title="RAG AI Assistant")
st.title("📄 RAG AI Chat Assistant")

# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
# Option 2: Agar specific PDF GitHub par hai
elif os.path.exists("Rag.pdf"):
    with st.spinner("Loading default PDF..."):
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
        
        st.session_state.chunks = sentence_chunk_text(text)
        st.info(f"📄 Loaded default PDF: {len(st.session_state.chunks)} chunks")

# Create embeddings and index
if st.session_state.chunks and st.session_state.embeddings is None:
    with st.spinner("Creating embeddings..."):
        # Load embedder
        @st.cache_resource
        def load_embedder():
            return SentenceTransformer("all-mpnet-base-v2")
        
        embedder = load_embedder()
        
        # Create embeddings
        chunk_embeddings = embedder.encode(st.session_state.chunks)
        chunk_embeddings = np.array(chunk_embeddings)
        
        # Normalize
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_embeddings = chunk_embeddings / norms
        
        st.session_state.embeddings = chunk_embeddings
        st.session_state.embedder = embedder
        
        # Create FAISS index if available
        if FAISS_AVAILABLE:
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(chunk_embeddings)
            st.session_state.index = index
            st.success("✅ FAISS index created")
        else:
            st.info("⚠️ Using numpy for similarity (slower but works)")

# Search function (FAISS or numpy)
def search_similar(query, k=3):
    if st.session_state.embeddings is None:
        return [], []
    
    # Get query embedding
    query_emb = st.session_state.embedder.encode([query])
    query_emb = np.array(query_emb)
    
    # Normalize
    query_norm = np.linalg.norm(query_emb)
    query_emb = query_emb / query_norm
    
    # Search using FAISS if available
    if FAISS_AVAILABLE and st.session_state.index is not None:
        distances, indices = st.session_state.index.search(query_emb, k)
        return distances[0], indices[0]
    
    # Fallback: numpy similarity
    else:
        similarities = np.dot(st.session_state.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_distances = similarities[top_indices]
        return top_distances, top_indices

# Groq client
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return None

client = get_groq_client()

if not client:
    st.error("⚠️ GROQ_API_KEY not found! Please add it in Streamlit secrets.")
    st.stop()

# Query input
query = st.text_input("🔍 Ask your question")

if query and st.session_state.chunks:
    with st.spinner("Searching and generating answer..."):
        # Search similar chunks
        distances, indices = search_similar(query, k=3)
        
        if len(indices) > 0:
            retrieved_chunks = [st.session_state.chunks[i] for i in indices]
            context = "\n\n".join(retrieved_chunks)
            
            # Create prompt
            prompt = f"""Using ONLY the context below, answer clearly in 3-4 sentences.

Context:
{context}

Question:
{query}"""
            
            # Get response from Groq
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",  # Better model name
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                confidence = round(float(distances[0]), 2) if len(distances) > 0 else 0
                
                # Display results
                st.subheader("🤖 AI Answer")
                st.write(answer)
                
                # Show sources
                with st.expander("📚 Source Chunks"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.write(f"**Chunk {i+1}** (score: {distances[i]:.3f})")
                        st.write(chunk[:300] + "...")
                        st.divider()
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Confidence: {confidence}")
                with col2:
                    if hasattr(response, 'usage'):
                        st.caption(f"Tokens: {response.usage.total_tokens}")
            
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

elif query and not st.session_state.chunks:
    st.warning("Please upload a PDF first!")
