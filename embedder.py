#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
embedder.py - SEC 10-K Document Embedder
================================================
SEC 10-K 문서 임베딩 및 FAISS 인덱싱 스크립트

Features:
- Used OpenAI text-embedding-3-large (3,072-d) embeddings
- Added to FAISS IndexFlatIP after L2 normalization(=Cosine Similarity)
- Saves as '.index' and '.meta.jsonl' files
- Chunk files are consolidated and saved in a single *_chunks.jsonl file

Usage
-----
    python embedder.py \
        --chunks_dir ./10k_chunked \
        --index_path ./faiss/large.index \
        --batch_size 64
"""

from __future__ import annotations
import argparse, json, pathlib, time
from typing import List, Generator

import numpy as np
import faiss, openai, tiktoken, tqdm

# ========= Configuration===========================
API_KEY      = "YOUR_API_KEY_HERE"  
MODEL        = "text-embedding-3-large"
DIM          = 3072
BATCH_SIZE   = 64          # 64×평균500tok ≈ 32k token < 8192 limit
SLEEP_SEC    = 0.2         # 5 req/sec → QPM 300 한도 안쪽
# ==================================================

openai.api_key = API_KEY
enc = tiktoken.get_encoding("cl100k_base")

def tok_len(txt: str) -> int:
    return len(enc.encode_ordinary(txt))

def batched(it: List[dict], n: int) -> Generator[List[dict], None, None]:
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == n:
            yield batch; batch = []
    if batch:
        yield batch

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = openai.embeddings.create(model=MODEL,
                                    input=texts,
                                    encoding_format="float")
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

# Process a single JSONL file and update the FAISS index
def process_file(fpath: pathlib.Path, index: faiss.Index, meta_list: List[dict]):
    with fpath.open() as fp:
        lines = [json.loads(l) for l in fp]
    for batch in batched(lines, BATCH_SIZE):
        texts = [rec["text"] for rec in batch]
        vecs = embed_texts(texts)
        index.add(vecs)
        meta_list.extend(batch)
        time.sleep(SLEEP_SEC)


def main(chunks_dir: str, index_path: str, batch_size: int):
    global BATCH_SIZE
    BATCH_SIZE = batch_size

    # Check if the directory exists
    files = list(pathlib.Path(chunks_dir).glob("*_chunks.jsonl"))
    if not files:
        raise FileNotFoundError("No *_chunks.jsonl files under", chunks_dir)

    index = faiss.IndexFlatIP(DIM)
    metadata: List[dict] = []

    for f in tqdm.tqdm(files, desc="embedding"):
        process_file(f, index, metadata)

    out_idx = pathlib.Path(index_path)
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_idx))

    meta_path = out_idx.with_suffix(".meta.jsonl")
    with meta_path.open("w") as fw:
        for rec in metadata:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ vectors : {index.ntotal:,}  saved → {out_idx}")
    print(f"✅ metadata: {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True, help="folder with *_chunks.jsonl")
    ap.add_argument("--index_path", required=True, help="output faiss index path")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    main(args.chunks_dir, args.index_path, args.batch_size)