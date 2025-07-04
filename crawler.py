"""
crawler.py - SEC 10-K Filing Crawling
=====================================
10-K HTML 파일 크롤링

Features:
- Fetches S&P 100 tickers from Wikipedia that are recently updated from the most famous S&P 100 ETF(Blackrock's S&P 100 ETF)
- Downloads 10-K filings from SEC EDGAR API
- Saves filings in structured directories in json format

Usage:
    python crawler.py
"""
import os, time, json, re
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime

# ================= Configuration ================
BASE_DIR   = "./10k_raw"        # change if you want a different root
USER_AGENT = "YOUR_EMAIL"   # <-- put real contact per SEC policy
START_YEAR, END_YEAR = 2023, 2025
# ================================================

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

# 1) fetch S&P 100 table 
sp100_url = "https://en.wikipedia.org/wiki/S%26P_100"        # Blackrock's S&P 100 ETF list
tables = pd.read_html(sp100_url, match="Symbol")[0]          # first table with “Symbol” col
tickers = sorted(tables["Symbol"].unique())                  # list of tickers
print(f"✅ Collected {len(tickers)} tickers from Wikipedia")

# 2) load SEC ticker ↔ CIK cross-walk
crosswalk_url = "https://www.sec.gov/files/company_tickers.json"
crosswalk_json = session.get(crosswalk_url, timeout=30).json()
mapping = { item["ticker"]: item["cik_str"] for item in crosswalk_json.values() }
missing = [t for t in tickers if t not in mapping]
if missing:
    print("⚠️ Tickers missing from cross-walk (will be skipped):", ", ".join(missing))
tickers = [t for t in tickers if t in mapping]               # drop if not found

# helper to build primary-doc URL 
def primary_doc_url(cik: str, accession: str, doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik):d}/{accession.replace('-', '')}/{doc}"

# 3) iterate over tickers 
os.makedirs(BASE_DIR, exist_ok=True)
for ticker in tqdm(tickers, desc="Downloading 10-Ks"):
    cik = f"{int(mapping[ticker]):010d}"                      # pad to 10 digits
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = session.get(sub_url, timeout=30).json()

    forms   = data["filings"]["recent"]["form"]
    dates   = data["filings"]["recent"]["filingDate"]
    access  = data["filings"]["recent"]["accessionNumber"]
    docs    = data["filings"]["recent"]["primaryDocument"]

    # scan recent filings list (ordered newest→oldest)
    downloads = []
    for f, d, a, doc in zip(forms, dates, access, docs):
        if f == "10-K" and START_YEAR <= int(d[:4]) <= END_YEAR:
            downloads.append((d, a, doc))

    if not downloads:
        continue

    # ensure company folder exists
    comp_dir = os.path.join(BASE_DIR, ticker)
    os.makedirs(comp_dir, exist_ok=True)

    for filing_date, accession, doc in downloads:
        url = primary_doc_url(cik, accession, doc)
        fname = f"{filing_date}_{doc}"
        fpath = os.path.join(comp_dir, fname)

        # skip if already downloaded
        if os.path.exists(fpath):
            continue

        try:
            html = session.get(url, timeout=30).content
            with open(fpath, "wb") as fp:
                fp.write(html)
        except Exception as e:
            print(f"❌ {ticker} {accession}: {e}")

        time.sleep(0.12)   # ~8 req/s – comply with SEC rate limit