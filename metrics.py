# metrics.py
# metrics.py
import nltk
from sentence_transformers import SentenceTransformer, util

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # ADD THIS LINE

# Load embedder
_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Download once (quietly)
nltk.download('punkt', quiet=True)

# Load lightweight embedder
_embedder = SentenceTransformer('all-MiniLM-L6-v2')

def bleu_score(reference: str, candidate: str) -> float:
    """BLEU score (higher = better)"""
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return nltk.translate.bleu_score.sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5))

def bertscore(reference: str, candidate: str) -> float:
    """Semantic similarity (0-1)"""
    ref_emb = _embedder.encode(reference, convert_to_tensor=True)
    cand_emb = _embedder.encode(candidate, convert_to_tensor=True)
    return util.cos_sim(ref_emb, cand_emb).item()

def evaluate(golden: str, generated: str) -> dict:
    return {
        "BLEU": round(bleu_score(golden, generated), 3),
        "BERTScore": round(bertscore(golden, generated), 3)
    }
