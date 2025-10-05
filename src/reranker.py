from sentence_transformers import CrossEncoder
from typing import List, Tuple

CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self, model_name: str = CROSS_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """
        candidates: list of texts
        повертає список (index_in_candidates, score) від найкращого до гіршого
        """
        pairs = [[query, c] for c in candidates]
        scores = self.model.predict(pairs)  # higher = better
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in idx_sorted]