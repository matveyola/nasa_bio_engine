# retriever.py
import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    """
    Пошук по існуючому FAISS-індексу.
    - Очікує, що індекс зібраний на L2-нормованих ембедингах і має IP-метрику (cosine).
    - Індекс збережений у index_path (faiss.write_index).
    - Позиції/ID, які повертає FAISS, відповідають рядкам df (або індексуйте через IndexIDMap із id=row_index).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "abstract",
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "docs.faiss",
        build_if_missing: bool = False,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.texts = self.df[text_col].fillna("").astype(str).tolist()
        self.model = SentenceTransformer(model_name, device=device)
        self.index_path = index_path
        self.batch_size = batch_size

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        elif build_if_missing:
            self.index = self._build_index()
            faiss.write_index(self.index, index_path)
        else:
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.d = self.index.d  # розмірність ембедів

    def _embed_norm(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # важливо для cosine через IP
            batch_size=self.batch_size,
            show_progress_bar=False
        ).astype("float32", copy=False)
        return embs

    def _build_index(self) -> faiss.Index:
        # точний пошук: IndexFlatIP + нормовані ембединги -> cosine similarity
        xb = self._embed_norm(self.texts)
        index = faiss.IndexFlatIP(xb.shape[1])
        # Якщо хочеш зберегти власні ID (напр. рядки df), обгорни:
        # index = faiss.IndexIDMap2(index); ids = np.arange(len(xb), dtype=np.int64); index.add_with_ids(xb, ids)
        index.add(xb)
        return index

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        q = self._embed_norm([query])  # (1, d)
        D, I = self.index.search(q, top_k)  # D — косинусні схожості
        I = I[0]; D = D[0]

        mask = I != -1
        idx = I[mask].tolist()
        scores = D[mask].tolist()

        hits = self.df.iloc[idx].copy()
        hits.insert(0, "score", np.round(scores, 6))
        hits.insert(1, "row_index", idx)
        return hits

    # опційно: додавання нових документів
    def add_documents(self, new_texts: List[str], save: bool = True):
        xb = self._embed_norm(new_texts)
        # якщо індекс — чистий IndexFlatIP:
        self.index.add(xb)
        if save:
            faiss.write_index(self.index, self.index_path)
