# app.py — FINAL CLEAN VERSION
import streamlit as st
import time
import os

st.set_page_config(page_title="AI-PM Toolkit", layout="wide")  # ONLY HERE

st.title("AI-PM Toolkit")

# === IMPORTS WITH ERROR HANDLING ===
try:
    from inference import generate
    INFERENCE_OK = True
except Exception as e:
    st.error(f"Inference failed: {e}")
    INFERENCE_OK = False

try:
    from metrics import evaluate
    METRICS_OK = True
except Exception as e:
    st.error(f"Metrics failed: {e}")
    METRICS_OK = False

try:
    from rag import extract_text_from_pdf, chunk_text, upsert_chunks, retrieve_chunks, rag_generate
    RAG_OK = True
except Exception as e:
    st.error(f"RAG failed: {e}")
    RAG_OK = False

# === TABS ===
tab1, tab2 = st.tabs(["A/B Prompt Lab", "RAG 101"])


# === TAB 1: A/B TESTING ===
with tab1:
    if not INFERENCE_OK:
        st.error("Fix inference.py first.")
    elif not METRICS_OK:
        st.warning("Metrics loading...")
    else:
        st.markdown("## A/B Prompt Testing")
        model = st.selectbox("LLM", ["Phi-3"], key="model_ab")
        
        col1, col2 = st.columns(2)
        with col1:
            prompt_a = st.text_area("Prompt A", "Tell me about Zohran Mamdani.")
            run_a = st.button("Run A")
        with col2:
            prompt_b = st.text_area("Prompt B", "Summarize Zohran Mamdani.")
            run_b = st.button("Run B")

        if run_a:
            with st.spinner("Generating A..."):
                resp_a, _ = generate(prompt_a, model)
                st.write("**Response A:**")
                st.write(resp_a)
                reference = "Zohran Mamdani is a democratic socialist and New York State Assembly member."
                scores = evaluate(resp_a, reference)
                st.json(scores)
                
        if run_b:
            with st.spinner("Generating B..."):
                resp_b, _ = generate(prompt_b, model)
                st.write("**Response B:**")
                st.write(resp_b)
                reference = "Zohran Mamdani is a democratic socialist and New York State Assembly member."
                scores = evaluate(resp_b, reference)
                st.json(scores)
# === TAB 2: RAG 101 ===
with tab2:
    if not RAG_OK:
        st.error("RAG failed to load.")
    else:
        st.markdown("## RAG: Upload & Query PDF")
        uploaded = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded and st.button("Index PDF"):
            with st.spinner("Indexing..."):
                text = extract_text_from_pdf(uploaded)
                chunks = chunk_text(text)
                upsert_chunks(chunks, uploaded.name)
            st.success(f"Indexed {len(chunks)} chunks")

        query = st.text_input("Ask:")
        if st.button("Generate") and query:
            with st.spinner("Retrieving..."):
                context = retrieve_chunks(query)
            with st.spinner("Generating..."):
                resp, meta = rag_generate(query, "Phi-3", context)
            st.markdown("### Answer")
            st.write(resp)
            st.caption(f"Generated in {meta['latency_sec']}s")
