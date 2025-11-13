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
        with st.spinner("Downloading Phi-3 GGUF (2.2GB)..."):
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                filename=hf_file,
                local_dir="."
            )
            os.rename(hf_file, model_path)
            st.success("Model ready!")

    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        n_batch=512,
        n_gpu_layers=0,
        verbose=False
    )
        
def generate(prompt: str, model):
    with st.spinner("Generating (3–5s)..."):
        # llama.cpp streaming returns dicts with 'choices' → text
        output = model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["<|end|>", "<|user|>", "<|assistant|>"],
            stream=True
        )
        
        response = ""
        placeholder = st.empty()
        for chunk in output:
            # CORRECT: chunk["choices"][0]["delta"]["content"] or ["text"]
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                response += text
                placeholder.markdown(f"**Answer**\n{response}")
        
        return response
