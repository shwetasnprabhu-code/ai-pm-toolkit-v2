from huggingface_hub import hf_hub_download
import os

print("Downloading Phi-3 GGUF (2.4GB)... This may take 2-10 minutes.")

hf_hub_download(
    repo_id="TheBloke/Phi-3-mini-4k-instruct-GGUF",
    filename="phi-3-mini-4k-instruct.Q4_0.gguf",
    local_dir=".",
    local_dir_use_symlinks=False,
    resume_download=True  # Resumes if interrupted
)

os.rename("phi-3-mini-4k-instruct.Q4_0.gguf", "phi3.gguf")
print("PHI-3 GGUF DOWNLOADED SUCCESSFULLY!")
print("File size:", os.path.getsize("phi3.gguf") / (1024**3), "GB")
