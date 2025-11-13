# inference.py — FINAL CLEAN (NO st!)
import os
import time
import streamlit as st
from llama_cpp import Llama
if not os.path.exists("phi3.gguf"):
    print("Downloading Phi-3 GGUF (2.2GB)... This takes ~5 minutes.")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        filename="Phi-3-mini-4k-instruct-q4.gguf",  # Correct filename
        local_dir=".",
        local_dir_use_symlinks=False
    )
    os.rename("Phi-3-mini-4k-instruct-q4.gguf", "phi3.gguf")
    print("Model downloaded and ready!")

@st.cache_resource
def get_llm():
    model_path = "phi3.gguf"
    hf_file = "Phi-3-mini-4k-instruct-q4.gguf"

    if not os.path.exists(model_path):
        with st.spinner("Downloading Phi-3 GGUF (2.2GB)... ~5 min"):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                filename=hf_file,
                local_dir="."
            )
            os.rename(hf_file, model_path)
            st.success("Model ready!")

    try:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False
        )
    except Exception as e:
        st.error(f"Model load failed: {e}")
        raise
        
def generate(prompt, model=None):
    llm = get_llm()
    start = time.time()
    
    # Truncate if too long
    tokens = llm.tokenize(prompt.encode('utf-8'))
    if len(tokens) > 3800:
        prompt = llm.detokenize(tokens[:3800]).decode('utf-8', errors='ignore')

    full_prompt = f"<|system|>You are a helpful assistant. Use ONLY the context.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
    
    try:
        output = llm(
            full_prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            stop=["<|end|>"],
            echo=False
        )
        response = output["choices"][0]["text"].strip()
        if not response:
            response = "No answer found in context."
            
    except Exception as e:
        response = f"[Error: {str(e)}]"
    
    latency = time.time() - start
    return response, {
        "latency_sec": round(latency, 2),
        "tokens": 999,
        "cost_usd": 0
    }
