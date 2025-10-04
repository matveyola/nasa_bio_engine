# -*- coding: utf-8 -*-
"""
Пошук по FAISS-індексу з MMR-diversity та прев'ю тексту.
Повертає також meta_idx (рядок у meta.jsonl), щоб легко тягнути повний текст.
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

STORE_DIR  = Path("store")
INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH  = STORE_DIR / "meta.jsonl"

# !!! ВАЖЛИВО: має збігатися з EMB_MODEL у build_index.py
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
# Напр., для багатомовного пошуку: "intfloat/multilingual-e5-small"


# ---------- helpers ----------
def load_meta() -> List[Dict]:
    metas: List[Dict] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def _mmr_select(
    q_vec: np.ndarray,              # shape (d,)
    cand_vecs: np.ndarray,          # shape (n, d), cosine (нормовані)
    cand_indices: List[int],        # індекси у метаданих/FAISS
    k: int,
    lambda_mult: float = 0.5
) -> List[int]:
    """
    Maximal Marginal Relevance: обирає k різноманітних і релевантних.
    Повертає підмножину cand_indices (довжини <= k) у порядку відбору.
    """
    n = cand_vecs.shape[0]
    if n == 0:
        return []

    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)

    # попередньо порахуємо схожість до запиту
    sim_to_q = cand_vecs @ q_vec  # shape (n,)

    # зберігаємо поточні максимум-сим до вже вибраних
    sim_to_sel = np.zeros(n, dtype=np.float32)

    for _ in range(min(k, n)):
        # MMR: λ*sim(q,d) - (1-λ)*max_sim(d, S)
        mmr_scores = lambda_mult * sim_to_q - (1 - lambda_mult) * sim_to_sel
        mmr_scores[selected_mask] = -1e9  # вже обрані не беремо
        j = int(np.argmax(mmr_scores))
        if mmr_scores[j] <= -1e8:
            break
        selected.append(j)
        selected_mask[j] = True
        # оновимо max схожість кожного кандидата до множини обраних
        sel_vec = cand_vecs[j]  # (d,)
        sim_js = cand_vecs @ sel_vec  # (n,)
        sim_to_sel = np.maximum(sim_to_sel, sim_js)

    # повертаємо оригінальні індекси meta за порядком відбору
    return [cand_indices[j] for j in selected]


def search(
    query: str,
    k: int = 6,
    mmr: bool = False,
    lambda_mult: float = 0.5,
    max_per_file: int = 2,
    return_text: bool = True,
) -> List[Dict]:
    """
    Повертає список результатів:
      {
        'rank', 'score', 'title', 'file', 'chunk_idx',
        'meta_idx', 'preview', ('text' якщо return_text=True)
      }
    """
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Немає індексу/метаданих. Спочатку запустіть build_index.py")

    # модель ембеддінгів (та сама, що й на етапі індексації)
    model = SentenceTransformer(EMB_MODEL)

    # читаємо індекс/метадані
    index = faiss.read_index(str(INDEX_PATH))
    metas = load_meta()

    # ембеддінг запиту (нормований)
    q = model.encode([query], normalize_embeddings=True).astype("float32")  # (1, d)
    d = q.shape[1]

    # базовий пошук
    topn = min(max(k * 10, k), index.ntotal)  # запас для MMR/дедупу
    scores, idxs = index.search(q, topn)      # scores: (1, topn); idxs: (1, topn)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    # при потребі — MMR (вимагає вектори документів; для IndexFlatIP вони є)
    if mmr and topn > 0:
        # реконструюємо вектори кандидатів
        cand_vecs = np.vstack([index.reconstruct(int(i)) for i in idxs]).astype("float32")
        q_vec = q[0]  # (d,)
        sel_meta_idxs = _mmr_select(q_vec, cand_vecs, idxs, k=k, lambda_mult=lambda_mult)
        # оновлюємо idxs/scores згідно з порядком MMR
        order = {mid: r for r, mid in enumerate(sel_meta_idxs)}
        # фільтруємо та сортуємо за mmr-порядком
        idxs = sel_meta_idxs
        scores = [float(index.search(q, 1)[0][0][0])] * len(idxs)  # скор не критичний при MMR
    else:
        # звужуємо до k
        idxs = idxs[:k]
        scores = scores[:k]

    # дедуп/ліміт по файлах
    out: List[Dict] = []
    per_file_count: Dict[str, int] = {}
    rank = 1
    for meta_idx in idxs:
        if meta_idx < 0 or meta_idx >= len(metas):
            continue
        m = metas[meta_idx]
        f = m.get("file", "")
        if max_per_file and per_file_count.get(f, 0) >= max_per_file:
            continue
        per_file_count[f] = per_file_count.get(f, 0) + 1

        text = m.get("text") or ""
        preview = text[:400].replace("\n", " ")
        item = {
            "rank": rank,
            "score": float(scores[rank-1]) if rank-1 < len(scores) else 0.0,
            "title": m.get("title"),
            "file": f,
            "chunk_idx": m.get("chunk_idx"),
            "meta_idx": meta_idx,
            "preview": preview,
        }
        if return_text:
            item["text"] = text
        out.append(item)
        rank += 1
        if len(out) >= k:
            break

    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="FAISS semantic search over local index")
    ap.add_argument("query", type=str, nargs="+", help="текст запиту")
    ap.add_argument("--k", type=int, default=6, help="кількість результатів")
    ap.add_argument("--mmr", action="store_true", help="включити MMR-diversity")
    ap.add_argument("--lambda", dest="lambda_mult", type=float, default=0.5,
                    help="MMR λ (0..1), більше = більше релевантність, менше = більше різноманіття")
    ap.add_argument("--max-per-file", type=int, default=2,
                    help="максимум чанків з одного файлу (0 = без ліміту)")
    ap.add_argument("--json", action="store_true", help="вивести JSON замість тексту")
    args = ap.parse_args()

    query = " ".join(args.query)
    try:
        hits = search(
            query=query,
            k=args.k,
            mmr=args.mmr,
            lambda_mult=args.lambda_mult,
            max_per_file=args.max_per_file,
            return_text=False,  # у CLI показуємо лише прев'ю
        )
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    if not hits:
        print("Нічого не знайдено.")
        return

    for h in hits:
        print(f"[{h['rank']}] score={h['score']:.3f} | {h['title']} (chunk {h['chunk_idx']}) | meta_idx={h['meta_idx']}")
        print(f"    {h['file']}")
        print(f"    {h['preview']}\n")


if __name__ == "__main__":
    main()
