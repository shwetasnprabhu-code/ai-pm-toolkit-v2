# metrics.py — FINAL WORKING VERSION
import nltk
import streamlit as st

# FORCE DOWNLOAD ON FIRST RUN
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # For METEOR
    return True

download_nltk()  # Run once

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score

def evaluate(response: str, reference: str = "ground truth summary") -> dict:
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(response.lower())
        
        bleu = sentence_bleu([ref_tokens], cand_tokens)
        meteor = single_meteor_score(" ".join(ref_tokens), " ".join(cand_tokens))
        
        return {
            "BLEU": round(bleu, 3),
            "METEOR": round(meteor, 3),
            "Length": len(cand_tokens)
        }
    except Exception as e:
        return {"Error": str(e)}
