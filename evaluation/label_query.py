#!/usr/bin/env python
# evaluation/label_query.py
"""
label_query.py — Labeling Queries with LLM
=======================================
한 쿼리에 대한 Similarity 상위 청크들이
쿼리에 대한 답변으로 사용될 수 있을지 여부를
OpenAI LLM을 사용하여 라벨링하는 스크립트

Features:
────────────────────────────────────────────
• Input
    - queries.csv: Query list to label (q, ticker, company_name)
    - chunk.index: FAISS index file (embedding vectors)
    - chunk.meta.jsonl: Chunk metadata (ticker, filing_year, part, section_item, section_title, text)
    - company_name.json: Ticker → Company Name mapping
• Query Processing
    - Embed each query and retrieve top-K similar chunks via FAISS
    - Evaluate each chunk's answerability using GPT-4o-mini (Y/N)
• Batch Processing
    - Group chunks in BATCH_SIZE(5) for LLM calls
    - Manage rate limits with sleep_sec delay between API calls
• Output: llm_labels.jsonl (JSONL format)
    - question: Original query
    - ticker: Target company ticker of the query
    - top_k: Number of retrieved chunks
    - chunks: [{vec_id, rank, sim, yes, ticker}, ...]

Usage:
    # Label single query
    python label_query.py --query_id 0
    
    # Label all queries
    python label_query.py --all
    
    # Resume from middle
    python label_query.py --all --start_from 100
"""
import argparse, json, os, sys, time
from typing import List, Dict

import faiss, openai, numpy as np, pandas as pd
from tqdm import tqdm

# Configuration
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL   = "gpt-4o-mini"
BATCH_SIZE  = 5
API_KEY = "YOUR_API_KEY_HERE"

SYSTEM_PROMPT = (
    "You are a helpful assistant who decides, for each numbered 10-K excerpt, "
    "whether it contains evidence to answer the question. "
    "Return **EXACTLY {n} letters** 'Y' or 'N', separated by a single '/'. "
    "DO NOT add spaces, extra letters or a trailing slash."
)

USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "EXCERPTS (1-{n}):\n{bullets}\n\n"
    "Respond with {n} letters 'Y' or 'N' separated by '/'."
)

def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_company_names(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)                 # { "WMT": "Walmart Inc.", ... }

def embedding(text: str) -> np.ndarray:
    return np.array(
        openai.embeddings.create(
            model=EMBED_MODEL, input=text
        ).data[0].embedding,
        dtype="float32",
    )

def faiss_search(index, vec: np.ndarray, k: int):
    dist, ids = index.search(vec[np.newaxis, :], k)
    return ids[0], dist[0]

def llm_batch(question: str,
              metas: List[Dict],
              company_map: Dict[str, str]) -> List[bool]:
    bullets = []
    for i, m in enumerate(metas):
        cname = company_map.get(m["ticker"], m["ticker"])
        header = (f"Excerpt from {cname} {m['filing_year']} 10-K, "
                  f"{m['part']} / {m['section_item']} – {m['section_title']}")
        bullets.append(f"{i+1}. {header}\n{m['text']}")
    user_msg = USER_TEMPLATE.format(question=question,
                                    n=len(bullets),
                                    bullets="\n\n".join(bullets))
    sys_msg  = SYSTEM_PROMPT.format(n=len(bullets))

    # API Call (with up to 3 retries)
    for attempt in range(3):
        try:
            res = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user",   "content": user_msg}],
                temperature=0,
            ).choices[0].message.content.strip()
            marks = [c.upper() for c in res.strip().strip("/").split("/") if c]
            marks = [c for c in marks if c in ("Y", "N")]
            if len(marks) < len(bullets):
                raise ValueError(f"Incomplete format: {res}")
            marks = marks[:len(bullets)]  
            return [c == "Y" for c in marks]
        except Exception as e:
            if attempt == 2: raise
            time.sleep(2 ** attempt)

# Main
# Label one query with LLM, and return JSON string
def label_one_query(args) -> str:
    openai.api_key = API_KEY

    # 1) Load Data
    queries   = pd.read_csv(args.queries_csv)
    question  = queries.at[args.query_id, "q"]
    ticker    = queries.at[args.query_id, "ticker"] if "ticker" in queries.columns else ""

    company_map = load_company_names("company_name.json")
    meta        = load_jsonl(args.meta_path)
    index       = faiss.read_index(args.index_path)

    # 2) Retrieve Top-K Chunks
    ids, sims = faiss_search(index, embedding(question), args.top_k)

    # 3) GPT Labeling
    results = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids  = ids[start:start+BATCH_SIZE]
        batch_meta = [meta[i] for i in batch_ids]
        flags      = llm_batch(question, batch_meta, company_map)

        for offset, vec_id in enumerate(batch_ids):
            idx = start + offset
            m   = batch_meta[offset]
            results.append({
                "vec_id": int(vec_id),
                "rank":   idx + 1,
                "sim":    float(sims[start + offset]),
                "yes":    flags[offset],
                "ticker": m["ticker"],
            })

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    # 4) JSON Return
    line_obj = {
        "question": question,
        "ticker":   ticker,
        "top_k":    args.top_k,
        "chunks":   results,
    }
    return json.dumps(line_obj, ensure_ascii=False)

# Main Execution
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="Label every row in queries_csv; append each JSON line to outfile")
    p.add_argument("--query_id", type=int, default=0,
                   help="Row index when --all is not used")
    p.add_argument("--start_from", type=int, default=0,
                   help="Start index when --all is used (for resuming)")
    p.add_argument("--top_k",      type=int, default=100)
    p.add_argument("--queries_csv",default="./queries.csv")
    p.add_argument("--index_path", default="../faiss/chunk.index")
    p.add_argument("--meta_path",  default="../faiss/chunk.meta.jsonl")
    p.add_argument("--outfile",    default="./llm_labels.jsonl")
    p.add_argument("--sleep_sec",  type=float, default=1.0,
                   help="Delay between GPT calls (seconds) - recommended: 1.0-2.0")
    args = p.parse_args()

    total_rows = len(pd.read_csv(args.queries_csv))
    
    if args.all:
        # Handle start_from in --all mode
        start_idx = max(0, min(args.start_from, total_rows - 1))
        id_list = range(start_idx, total_rows)
        print(f"Processing queries from index {start_idx} to {total_rows-1} (total: {len(id_list)} queries)")
    else:
        id_list = [args.query_id]

    # File mode: use 'a' (append) if start_from is not 0 or file exists, otherwise use 'w' (write)
    mode = "a" if (args.all and args.start_from > 0) or (os.path.exists(args.outfile) and args.all) else "w"
    
    with open(args.outfile, mode, encoding="utf-8") as fout:
        # Show tqdm progress bar when using --all option, keep original logging for single query
        if args.all:
            progress_bar = tqdm(id_list, desc="Processing queries", unit="query")
            for qid in progress_bar:
                progress_bar.set_postfix({"current": qid, "sleep": f"{args.sleep_sec}s"})
                args.query_id = qid
                try:
                    json_line = label_one_query(args)
                    fout.write(json_line + "\n")
                    fout.flush()
                except Exception as e:
                    tqdm.write(f"[{qid}] ⚠️  error → {e}")
        else:
            # Keep original logging style for single query processing
            for qid in id_list:
                args.query_id = qid
                try:
                    json_line = label_one_query(args)
                    fout.write(json_line + "\n")
                    fout.flush()
                    print(f"[{qid}] ✅  done")
                except Exception as e:
                    print(f"[{qid}] ⚠️  error → {e}")