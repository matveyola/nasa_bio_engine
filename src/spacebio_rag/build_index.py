# -*- coding: utf-8 -*-
"""
Побудова локального векторного індексу для RAG (потоково, без OOM).
Підтримує: .pdf, .txt/.md, .docx, .html/.htm, .csv/.tsv
Зберігає: store/index.faiss та store/meta.jsonl

Залежності (у вашому venv):
  pip install faiss-cpu sentence-transformers pypdf docx2txt beautifulsoup4 lxml pandas tqdm numpy
"""

from __future__ import annotations
import os, re, json, uuid, hashlib, csv, sys
from pathlib import Path
from typing import List, Dict, Optional, Iterable

import numpy as np
import faiss
from tqdm import tqdm
from pypdf import PdfReader
import docx2txt
from bs4 import BeautifulSoup
import pandas as pd
from pandas.errors import ParserError
from sentence_transformers import SentenceTransformer

# ---------- НАЛАШТУВАННЯ ----------
DATA_DIR   = Path("data")
STORE_DIR  = Path("store"); STORE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH  = STORE_DIR / "meta.jsonl"

EMB_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 160
EMB_BATCH      = 256  # скільки чанків ембедимо за раз

# (опц.) які саме колонки брати з конкретних CSV
TEXT_COLS_BY_FILENAME: Dict[str, List[str]] = {
    # "articles_with_full_text_20251004_104106.csv": ["title", "abstract", "full_text"]
}

# ---------- УТИЛІТИ ----------
_WS = re.compile(r"\s+", re.U)
def _norm(s: str) -> str:
    # швидка нормалізація пробілів по мірі надходження
    return _WS.sub(" ", s)

class StreamChunker:
    """Потокове чанкування: feed(text) -> yield готові шматки, flush() -> залишок."""
    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap
        self.buf: str = ""

    def feed(self, piece: str) -> Iterable[str]:
        if not piece:
            return
        # нормалізуємо лише додану частину та приєднуємо
        self.buf += _norm(piece)
        # віддаємо повні чанки
        while len(self.buf) >= self.size:
            yield self.buf[: self.size]
            self.buf = self.buf[self.size - self.overlap :]

    def flush(self) -> Iterable[str]:
        if self.buf:
            yield self.buf
            self.buf = ""

# ---------- РІДЕРИ ФАЙЛІВ ----------
def read_txt(path: Path) -> Iterable[str]:
    # читаємо файлом, але віддаємо одним шматком (далі все одно піде через StreamChunker)
    yield path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> Iterable[str]:
    with open(path, "rb") as f:
        r = PdfReader(f)
        for pg in r.pages:
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            yield txt

def read_docx(path: Path) -> Iterable[str]:
    yield docx2txt.process(str(path)) or ""

def read_html(path: Path) -> Iterable[str]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    yield soup.get_text(separator=" ")

def _bump_csv_field_limit():
    # підвищуємо ліміт на довжину одного поля CSV (для дуже довгих full_text)
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

def stream_csv_text(path: Path,
                    text_cols: Optional[List[str]] = None,
                    sep: Optional[str] = None,
                    encoding_hint: Optional[str] = None,
                    chunksize: int = 4000) -> Iterable[str]:
    """
    Потоково повертає порції тексту з CSV.
    Кожна порція — це кілька рядків, склеєних у великий текстовий блок.
    """
    _bump_csv_field_limit()

    def reader(enc: str, sep_val: Optional[str]):
        return pd.read_csv(
            path,
            encoding=enc,
            sep=sep_val if sep_val is not None else None,  # auto-infer якщо None
            dtype=str,
            on_bad_lines="skip",
            chunksize=chunksize,
            engine="python",   # python engine підтримує великий field_size_limit
        )

    it = None
    # 1) пробуємо з підказаним encoding або 'utf-8'
    encoders = [encoding_hint] if encoding_hint else []
    encoders += ["utf-8", "cp1251"]

    tried = []
    for enc in encoders:
        if enc is None: 
            continue
        try:
            it = reader(enc, sep)
            break
        except (UnicodeDecodeError, ParserError) as e:
            tried.append((enc, str(e)))
            it = None

    # 2) якщо не вдалось — пробуємо різні сепаратори
    if it is None:
        for enc in ["utf-8", "cp1251"]:
            for sep_candidate in (",", ";", "\t", "|"):
                try:
                    it = reader(enc, sep_candidate)
                    sep = sep_candidate  # зафіксуємо
                    break
                except Exception:
                    it = None
            if it is not None:
                break

    if it is None:
        raise RuntimeError(f"Не вдалося прочитати CSV: {path.name}; спроби={tried}")

    for df in it:
        # обираємо колонки
        if text_cols:
            cols = [c for c in text_cols if c in df.columns]
        else:
            cols = list(df.select_dtypes(include=["object"]).columns) or list(df.columns)
        if not cols:
            continue

        df = df[cols].fillna("")
        # зробимо одну велику порцію з цієї партії рядків
        lines = df.apply(lambda r: " | ".join(f"{c}: {r[c]}" for c in cols if r[c]),
                         axis=1).tolist()
        yield "\n".join(lines)

def load_file_stream(path: Path) -> Iterable[str]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:   return read_txt(path)
    if ext == ".pdf":            return read_pdf(path)
    if ext == ".docx":           return read_docx(path)
    if ext in {".html", ".htm"}: return read_html(path)
    if ext == ".csv" or path.name.lower().endswith(".tsv"):
        text_cols = TEXT_COLS_BY_FILENAME.get(path.name)
        # якщо точно знаєте, що CSV розділяється «;» — можна примусово:
        # return stream_csv_text(path, text_cols=text_cols, sep=";")
        sep = "\t" if path.name.lower().endswith(".tsv") else None
        return stream_csv_text(path, text_cols=text_cols, sep=sep)
    # інші типи ігноруємо
    return []

# ---------- ПОБУДОВА ІНДЕКСУ (ПОТОКОВО) ----------
def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Сканую:", DATA_DIR.resolve())
    files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    print("Знайдено у data/:", [(p.name, p.suffix.lower()) for p in files])
    if not files:
        print("У папці data/ немає файлів.")
        return

    model = SentenceTransformer(EMB_MODEL)
    index: Optional[faiss.IndexFlatIP] = None

    # готуємо meta.jsonl для покрокового запису
    meta_fp = META_PATH.open("w", encoding="utf-8")

    # невеликий буфер для батчового ембеддингу
    batch_texts: List[str] = []
    batch_metas:  List[Dict] = []
    total_chunks = 0

    def flush_batch():
        nonlocal index, batch_texts, batch_metas
        if not batch_texts:
            return
        embs = model.encode(
            batch_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")
        if index is None:
            index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        for m in batch_metas:
            meta_fp.write(json.dumps(m, ensure_ascii=False) + "\n")
        batch_texts.clear()
        batch_metas.clear()

    for path in tqdm(files, desc="Індексуємо файли"):
        stream = load_file_stream(path)
        if not stream:
            print(f"[SKIP] {path.name} — невідомий/порожній формат")
            continue

        chunker = StreamChunker(CHUNK_SIZE, CHUNK_OVERLAP)

        for piece in stream:
            # потоково ріжемо piece на чанки
            for ch in chunker.feed(piece):
                total_chunks += 1
                uid = str(uuid.uuid4())
                batch_texts.append(ch)
                batch_metas.append({
                    "id": uid,
                    "file": str(path),
                    "title": path.stem,
                    "chunk_idx": total_chunks,  # глобальний лічильник ок
                    "text": ch,
                })
                if len(batch_texts) >= EMB_BATCH:
                    flush_batch()

        # добираємо «хвіст» після файлу
        for ch in chunker.flush():
            total_chunks += 1
            uid = str(uuid.uuid4())
            batch_texts.append(ch)
            batch_metas.append({
                "id": uid,
                "file": str(path),
                "title": path.stem,
                "chunk_idx": total_chunks,
                "text": ch,
            })
            if len(batch_texts) >= EMB_BATCH:
                flush_batch()

    # фінальний флаш
    flush_batch()
    meta_fp.close()

    if index is None or index.ntotal == 0:
        print("Нема тексту для індексації.")
        return

    faiss.write_index(index, str(INDEX_PATH))
    print("OK ✅  Індекс збережено.")
    print(f" • Векторів: {index.ntotal}")
    print(f" • Індекс:   {INDEX_PATH}")
    print(f" • Метадані: {META_PATH}")

if __name__ == "__main__":
    build_index()
