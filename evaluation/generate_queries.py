#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_queries.py — query CSV (q, ticker, company_name) Generator
=======================================
회사별 질문을 탬플릿에 기반하여 생성하는 스크립트

Features:
────────────────────────────────────────────
• Input
    - company_by_sector.json: Company Group by Sector
    - templates_by_sector.json: Query Templates
    - company_name.json: Ticker → Company Name
• Query Sampling
    - Query Set per Company: Three General and Two Sector-specific Queries
• Format
    - q: "{company_name}의 {template_body_without_ellipsis}"
    - E.g.: "{Apple Inc.}의 {매출을 부문별로 설명해 줘.}"
• Output: q, ticker, company_name (CSV header included)

Usage:
    python generate_queries.py
"""

import argparse, csv, json, pathlib, random, re, sys

# ────────────────────────────────────────────────────────────
ELL = re.compile(r"^\.\.\.\s*")   # 앞쪽 "... " 제거 

# CLI Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--sector_json",   default="company_by_sector.json")
parser.add_argument("--template_json", default="templates_by_sector.json")
parser.add_argument("--name_json",     default="company_name.json")
parser.add_argument("--out",           default="queries.csv")
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# Data Load
sector_map   = json.load(open(args.sector_json))
templates_all= json.load(open(args.template_json))
name_map     = json.load(open(args.name_json))

gen_templates = templates_all["General"]

# Query Generation
rows = []
for sector, tics in sector_map.items():
    sec_tpls = templates_all.get(sector, [])
    for tic in tics:
        cname = name_map.get(tic, tic)
        # Template Selection
        if sector == "Others":
            chosen = random.sample(gen_templates, 5)
        else:
            g3 = random.sample(gen_templates, 3)
            if len(sec_tpls) >= 2:
                s2 = random.sample(sec_tpls, 2)
            else:  # 부족하면 General 로 보충
                s2 = random.sample(gen_templates, 2)
            chosen = g3 + s2
        # Generate Queries from Templates
        for tpl in chosen:
            body = ELL.sub("", tpl).strip()
            q = f"{cname}의 {body}"
            rows.append((q, tic, cname))

# Save to CSV
out_path = pathlib.Path(args.out)
with out_path.open("w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp)
    w.writerow(["q", "ticker", "company_name"])
    w.writerows(rows)
print(f"✅  {len(rows)} rows written → {out_path}")
