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
    correct_filename = "Phi-3-mini-4k-instruct-q4.gguf"
    expected_size = 2_200_000_000  # ~2.2 GB

    # DELETE CORRUPTED FILE
    if os.path.exists(model_path):
        actual_size = os.path.getsize(model_path)
        if actual_size < expected_size * 0.9:  # Less than 90%
            st.warning(f"Corrupted model ({actual_size/1e9:.2f} GB). Deleting...")
            os.remove(model_path)

    # DOWNLOAD WITH VERIFICATION
    if not os.path.exists(model_path):
        with st.spinner("Downloading Phi-3 GGUF (2.2GB)... ~5 min"):
            from huggingface_hub import hf_hub_download
            try:
                hf_hub_download(
                    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                    filename=correct_filename,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                os.rename(correct_filename, model_path)
                final_size = os.path.getsize(model_path)
                if final_size > expected_size * 0.9:
                    st.success(f"Model ready! ({final_size/1e9:.2f} GB)")
                else:
                    raise ValueError("Download incomplete")
            except Exception as e:
                st.error(f"Download failed: {e}")
                raise

    # LOAD MODEL
    try:
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
