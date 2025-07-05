#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reranker.py
==========================================================
쿼리에서 회사를 추출하고 해당 ticker의 청크에 가산점을 부여하는 RAG 시스템

Features:
- Extracts company tickers from user queries
- Retrieves top K1 chunks using FAISS vector search
- Applies bonus scores for chunks matching query tickers
- Reranks top K2 candidates using BGE reranker model
- Generates answers using OpenAI LLM with context from top M chunks
"""
import os, json, time, hashlib, faiss, openai, numpy as np, tiktoken, torch, pathlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= Configuration===========================
K1 = 300       # Number of chunks to retrieve from FAISS
K2 = 100       # Number of chunks to rerank
M = 10         # Number of chunks to return to the LLM
BONUS = 0.2    # Bonus to chunks matching query tickers
MODEL = "gpt-4o"
API_KEY = "YOUR_API_KEY_HERE"
EMB_MODEL = "text-embedding-3-large"
# ==================================================

ROOT = pathlib.Path(__file__).resolve().parent
IDX_PATH = str(ROOT / "faiss" / "chunk.index")
META_PATH = str(ROOT / "faiss" / "chunk.meta.jsonl")
COMPANY_NAME_PATH = str(ROOT / "evaluation" / "company_name.json")
SECTOR_PATH = str(ROOT / "evaluation" / "company_by_sector.json")

# Load FAISS index and metadata
openai.api_key = API_KEY
enc = tiktoken.get_encoding("cl100k_base")
index = faiss.read_index(IDX_PATH)
meta = [json.loads(l) for l in open(META_PATH)]
for i, rec in enumerate(meta):
    rec["vec_id"] = i

# Load company names and sector data
with open(COMPANY_NAME_PATH, 'r') as f:
    company_names = json.load(f)
    
with open(SECTOR_PATH, 'r') as f:
    company_by_sector = json.load(f)

# Create a reverse mapping from company names to tickers
name_to_ticker = {v.lower(): k for k, v in company_names.items()}

tok_rer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
mdl_rer = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-base").eval()

def embed(text: str) -> np.ndarray:
    vec = openai.embeddings.create(model=EMB_MODEL,
                                   input=text,
                                   encoding_format="float").data[0].embedding
    v = np.array(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def extract_tickers_from_query(query: str) -> list:
    prompt = f"""Extract company tickers from the following query. 
    If company names are mentioned, convert them to their stock tickers.
    Return ONLY a JSON array of tickers, nothing else.
    
    Available companies and their tickers:
    {json.dumps(company_names, indent=2)}
    
    Query: {query}
    
    Example output: ["AAPL", "MSFT"]
    If no companies are found, return: []
    """
    
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        tickers = json.loads(resp.choices[0].message.content)

        # Filter out invalid tickers
        valid_tickers = [t for t in tickers if t in company_names]
        return valid_tickers
    except:
        return []

def cite_header(rec: dict) -> str:
    parts = [rec["ticker"],
             str(rec["filing_year"]),
             rec["part"],
             rec.get("section_item", ""),
             rec.get("section_title", "")]
    return " ".join(f for f in parts if f).strip()

def retrieve(query: str, status_callback=None):
    # 1. Extract tickers from query
    if status_callback:
        status_callback("🏢 Extracting companies from query...", progress=20)
    query_tickers = extract_tickers_from_query(query)
    
    # Status callback for ticker detection
    if status_callback:
        if query_tickers:
            status_callback(f"✅ Detected companies: {', '.join(query_tickers)}", progress=20, status_type="success")
        else:
            status_callback("⚠️ No specific companies detected, searching all documents...", progress=20, status_type="warning")
    
    # 2. Vector Search
    if status_callback:
        status_callback(f"🔍 Searching top {K1} chunks...", progress=40)
    v = embed(query)
    D, I = index.search(v, K1)
    
    # 3. Apply additional cosine similarity score based on query tickers
    if status_callback:
        if query_tickers:
            status_callback(f"📊 Selecting documents for the detected company...", progress=60)
        else:
            status_callback(f"📊 No prioritizing for documents related to your query...", progress=60)
    
    scores_with_bonus = []
    for d, i in zip(D[0], I[0]):
        score = d  # cosine similarity
        # Add bonus if the chunk's ticker matches any in the query
        if meta[i]["ticker"] in query_tickers:
            score += BONUS
        scores_with_bonus.append((score, i))
    
    # 4. Filter Top K2 candidates after applying ticker bonus
    cand_sorted = sorted(scores_with_bonus, reverse=True)
    top_k2_candidates = cand_sorted[:K2]  # 상위 K2개만 선택
    cand_indices = [i for _, i in top_k2_candidates]
    
    # 5. Rerank the top K2 candidates and select M best chunks
    if status_callback:
        status_callback(f"🎯 Reranking top {K2} chunks to find best {M}...", progress=80)
    pairs = [query + " [SEP] " + meta[i]["text"] for i in cand_indices]
    toks = tok_rer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = mdl_rer(**toks).logits.squeeze(-1).tolist()
    
    reranked = sorted(zip(scores, cand_indices), reverse=True)
    result_ids = [i for _, i in reranked][:M]
    
    return [meta[i] for i in result_ids]

PROMPT = """### ROLE
You are a professional analyst of corporate filings and other official disclosures.

### CONTEXT
{ctx}

### QUESTION
{q}

### INSTRUCTIONS
* When citing, prefix the passage with metadata like [AAPL 2024 Part I Item 1 Business].
* Omit empty elements (often section_title).
* If the answer is not present, reply: "Not found in the provided excerpts."
* Respond concisely in fluent English.
"""

def answer(q: str, status_callback=None):
    recs = retrieve(q, status_callback)
    if not recs:
        return "Not found in the provided excerpts."
    
    if status_callback:
        status_callback(f"🤖 Generating answer from {len(recs)} best chunks...")
    
    ctx_blocks = [f"[{cite_header(r)}]\n{r['text']}" for r in recs]
    prompt = PROMPT.format(ctx="\n\n".join(ctx_blocks), q=q)
    
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content