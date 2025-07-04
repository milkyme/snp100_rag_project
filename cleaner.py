#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleaner.py - SEC 10-K Filing Cleaner
=====================================
10-K HTML 파일 클리닝 스크립트

Features:
- HTML to JSONL conversion with section extraction
- Two-stage cleaning pipeline (HTML parsing + text refinement)
- Support for iXBRL format

Usage:
    python cleaner.py --in_dir ./10k_raw --out_dir ./10k_cleaned
"""

import os
import re
import json
import argparse
import warnings
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from tqdm import tqdm

# SEC 10-K filing cleaner with two-stage processing pipeline"""
class FilingCleaner:
    
    # Regex patterns for part/item detection - more flexible patterns
    PART_PATTERN = re.compile(
        r"^\s*part\s+([ivxlcdm]+|\d+)\s*\.?\s*$|"  # "Part I" or "Part 1"
        r"^\s*part\s+([ivxlcdm]+|\d+)\s*[-–—]\s*|"  # "Part I -" or "Part I —"
        r"^\s*part\s+([ivxlcdm]+|\d+)\s*:\s*", 
        re.I | re.M
    )
    ITEM_PATTERN = re.compile(
        r"^\s*item\s+(\d+[a-z]?)\s*\.?\s*[-–—:]?\s*(.*)$",  # More flexible item pattern
        re.I | re.M
    )
    
    # Regex patterns for second-stage cleaning
    RE_EMPTY = re.compile(r"^(none\.?|not applicable\.?|n/a)$", re.I)
    RE_FOOTER = re.compile(r".*form\s+10-k.*\|\s*\d+$", re.I)
    RE_FOOTER2 = re.compile(r"^-?\s*\d+\s*-?$")  # Page numbers like "5" or "- 5 -"
    RE_HEADER = re.compile(r"^(table of contents|part\s+[ivxlcdm]+.*continued)$", re.I)
    RE_SAFE_HARBOR = re.compile(r"forward[- ]looking statements", re.I)
    
    SAFE_HARBOR_MAX_LINES = 80
    MIN_SECTION_LENGTH = 15
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Suppress BeautifulSoup XML warning for iXBRL docs
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    
    # ========== Stage 1: HTML Processing ==========
    
    # Convert HTML table to tab-delimited text
    def flatten_table(self, tag) -> str:
        rows = []
        for tr in tag.find_all("tr", recursive=False):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append("\t".join(cells))
        return "\n".join(rows)
    
    # Remove iXBRL tags while preserving text content
    def unwrap_ixbrl(self, soup):
        for tag in soup.find_all(True):
            if tag.name.startswith(("ix:", "ix", "xbrli")):
                tag.unwrap()
    
    # Clean HTML and perform initial parsing
    def clean_html(self, html_bytes: bytes) -> BeautifulSoup:
        soup = BeautifulSoup(html_bytes, "lxml")
        
        # Remove scripts and styles
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Handle iXBRL format
        self.unwrap_ixbrl(soup)
        
        # Flatten tables to text
        for table in soup.find_all("table"):
            table.replace_with(NavigableString(self.flatten_table(table)))
        
        # Convert <br> to newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")
        
        return soup
    
    # Extract Part I and Part II sections with their items
    def extract_parts(self, soup: BeautifulSoup) -> Dict[str, List[Tuple]]:
        parts = {}
        current_part = None
        current_item = None
        buffer = []
        
        # Roman numeral conversion
        roman_map = {
            "I": "I", "II": "II", "III": "III", "IV": "IV",
            "1": "I", "2": "II", "3": "III", "4": "IV"
        }
        
        def flush_buffer():
            nonlocal buffer, current_item, current_part
            if current_part and current_item and buffer:
                text = "\n".join(buffer).strip()
                if text:  # Only add non-empty sections
                    parts.setdefault(current_part, []).append((*current_item, text))
            buffer = []
        
        # Process all text nodes
        text_nodes = list(soup.find_all(string=True))
        
        if self.verbose and len(text_nodes) > 0:
            print(f"[DEBUG] Total text nodes: {len(text_nodes)}")
            # Print first few text nodes for debugging
            sample_texts = [str(node).strip()[:100] for node in text_nodes[:20] if str(node).strip()]
            print(f"[DEBUG] First few text samples:")
            for i, txt in enumerate(sample_texts[:5]):
                print(f"  {i}: {txt}")
        
        for node in text_nodes:
            text = str(node).replace("\xa0", " ")  # Convert nbsp to space
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            
            # Check for Part heading
            part_match = self.PART_PATTERN.match(text)
            if part_match:
                flush_buffer()
                # Try to extract part number from any group
                part_num = None
                for group in part_match.groups():
                    if group:
                        part_num = group.upper()
                        break
                
                if part_num and part_num in roman_map:
                    normalized_part = f"Part {roman_map[part_num]}"
                    current_part = normalized_part
                    current_item = None
                    if self.verbose:
                        print(f"[DEBUG] Found {current_part} from text: '{text}'")
                continue
            
            # Also check if the text contains "Part I" or "Part II" anywhere
            if not current_part:
                lower_text = text.lower()
                if "part i" in lower_text and "part ii" not in lower_text:
                    current_part = "Part I"
                    current_item = None
                    if self.verbose:
                        print(f"[DEBUG] Found Part I in text: '{text[:100]}'")
                elif "part ii" in lower_text and "part iii" not in lower_text:
                    current_part = "Part II"
                    current_item = None
                    if self.verbose:
                        print(f"[DEBUG] Found Part II in text: '{text[:100]}'")
            
            # Check for Item heading (only within Part I or II)
            if current_part in ("Part I", "Part II"):
                item_match = self.ITEM_PATTERN.match(text)
                if item_match:
                    flush_buffer()
                    item_num, title = item_match.groups()
                    # Clean up item number
                    item_num = re.sub(r'[^\d\w]', '', item_num).upper()
                    current_item = (f"Item {item_num}", title.strip())
                    if self.verbose:
                        print(f"[DEBUG]  └─ {current_item[0]}: {title.strip()[:50]}...")
                    continue
                
                # Also check for items without strict pattern matching
                lower_text = text.lower()
                if lower_text.startswith("item ") and len(text) < 200:
                    # Try to extract item number
                    item_match2 = re.search(r"item\s+(\d+[a-z]?)", lower_text)
                    if item_match2:
                        flush_buffer()
                        item_num = item_match2.group(1).upper()
                        # Extract title (everything after item number)
                        title_start = item_match2.end()
                        title = text[title_start:].strip(' .-–—:')
                        current_item = (f"Item {item_num}", title)
                        if self.verbose:
                            print(f"[DEBUG]  └─ {current_item[0]}: {title[:50]}... (loose match)")
                        continue
            
            # Accumulate text for current section
            if current_part in ("Part I", "Part II"):
                buffer.append(text)
        
        flush_buffer()
        
        if self.verbose:
            print(f"[DEBUG] Total parts extracted: {list(parts.keys())}")
            for part, items in parts.items():
                print(f"[DEBUG] {part}: {len(items)} items")
                for item_id, title, text, in items[:2]:  # Show first 2 items
                    print(f"[DEBUG]   - {item_id}: {title[:50]}... ({len(text)} chars)")
        
        return parts
    
    # Calculate relative weights for each section based on text length
    def compute_weights(self, items: List[Tuple]) -> List[Tuple]:
        lengths = [len(item[-1]) for item in items]
        total_length = sum(lengths) or 1
        
        weighted_items = []
        for (item_id, title, text), length in zip(items, lengths):
            weight = length / total_length
            weighted_items.append((item_id, title, text, weight))
        
        return weighted_items
    
    # ========== Stage 2: Text Cleaning ==========
    
    # Apply second-stage text cleaning rules
    def cleanse_text(self, text: str) -> str:
        if not text:
            return ""
        
        lines = text.splitlines()
        cleaned_lines = []
        skip_safe_harbor = False
        safe_harbor_line_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Handle empty lines
            if not line_stripped:
                if skip_safe_harbor and safe_harbor_line_count < self.SAFE_HARBOR_MAX_LINES:
                    skip_safe_harbor = False
                cleaned_lines.append(line_stripped)
                continue
            
            # Check for safe harbor statement block
            if self.RE_SAFE_HARBOR.search(line_stripped):
                skip_safe_harbor = True
                safe_harbor_line_count = 0
                continue
            
            if skip_safe_harbor:
                safe_harbor_line_count += 1
                continue
            
            # Skip footers, headers, and page numbers
            if (self.RE_FOOTER.match(line_stripped) or 
                self.RE_FOOTER2.match(line_stripped) or 
                self.RE_HEADER.match(line_stripped)):
                continue
            
            cleaned_lines.append(line_stripped)
        
        # Join non-empty lines
        return "\n".join([line for line in cleaned_lines if line])
    
    # Check if section should be skipped based on content
    def should_skip_section(self, text: str) -> bool:
        
        text = text.strip()
        # Skip if too short and matches empty pattern
        if len(text) <= self.MIN_SECTION_LENGTH and self.RE_EMPTY.match(text):
            return True
        return False
    
    # ========== File Processing ==========
    
    # Process a single HTML file through the complete pipeline
    def process_file(self, input_path: Path, output_dir: Path) -> int:
        # Extract metadata from filename: YYYY-MM-DD_ticker-xxxx.htm[l]
        match = re.match(r"(\d{4})-\d{2}-\d{2}_([a-z.]+)-", input_path.name, re.I)
        if not match:
            if self.verbose:
                print(f"[SKIP] Invalid filename pattern: {input_path.name}")
            return 0
        
        filing_year = match.group(1)
        ticker = match.group(2).upper()
        
        if self.verbose:
            print(f"\n[PROCESSING] {input_path.name}")
            print(f"  Ticker: {ticker}, Year: {filing_year}")
        
        # Stage 1: HTML processing
        try:
            with input_path.open("rb") as f:
                soup = self.clean_html(f.read())
        except Exception as e:
            print(f"[ERROR] Failed to parse {input_path.name}: {e}")
            return 0
        
        parts = self.extract_parts(soup)
        if not parts:
            print(f"[SKIP] No Part I/II found in {input_path.name}")
            return 0
        
        # Prepare output
        output_file = output_dir / f"{ticker}_{filing_year}.jsonl"
        records_written = 0
        records = []
        
        # Process each part and item
        for part_name in ("Part I", "Part II"):
            if part_name not in parts:
                if self.verbose:
                    print(f"  {part_name} not found")
                continue
            
            if self.verbose:
                print(f"  Processing {part_name} with {len(parts[part_name])} items")
            
            weighted_items = self.compute_weights(parts[part_name])
            
            for item_id, title, text, weight in weighted_items:
                # Stage 2: Text cleaning
                cleaned_text = self.cleanse_text(text)
                
                # Skip empty or invalid sections
                if not cleaned_text or self.should_skip_section(cleaned_text):
                    if self.verbose:
                        print(f"    Skipping {item_id} - empty or invalid")
                    continue
                
                record = {
                    "ticker": ticker,
                    "filing_year": int(filing_year),
                    "part": part_name,
                    "section_item": item_id,
                    "section_title": title,
                    "section_weight": round(weight, 4),
                    "text": textwrap.dedent(cleaned_text)
                }
                
                records.append(record)
                records_written += 1
                
                if self.verbose:
                    print(f"    Added {item_id}: {title[:30]}... ({len(cleaned_text)} chars)")
        
        # Write output
        if records:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write final cleaned version
            with output_file.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            if self.verbose:
                print(f"  Wrote {records_written} sections to {output_file}")
        else:
            if self.verbose:
                print(f"  No valid sections found after cleaning")
        
        return records_written


def main():
    parser = argparse.ArgumentParser(
        description="Clean SEC 10-K HTML filings and extract structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
            # Basic usage
            python cleaner.py --in_dir ./10k_raw --out_dir ./10k_cleaned
            
            # Debug mode - check why files aren't being processed
            python cleaner.py --in_dir ./10k_raw --out_dir ./10k_cleaned --limit 1 --verbose --debug
        """)
    )
    
    parser.add_argument("--in_dir", required=True, help="Directory containing raw HTML files")
    parser.add_argument("--out_dir", required=True, help="Output directory for cleaned JSONL files")
    parser.add_argument("--limit", type=int, help="Process only first N files (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (shows HTML structure)")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.in_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HTML files
    html_files = list(input_dir.rglob("*.htm*"))
    if args.limit:
        html_files = html_files[:args.limit]
    
    if not html_files:
        print(f"❌ No HTML files found in {input_dir}")
        return
    
    print(f"📁 Found {len(html_files)} HTML files in {input_dir}")
    print(f"📝 Output directory: {output_dir}")
    if args.verbose:
        print(f"🔍 Verbose mode enabled")
    
    # Debug mode - analyze first file
    if args.debug and html_files:
        print(f"\n🐛 Debug mode - analyzing first file: {html_files[0].name}")
        with html_files[0].open("rb") as f:
            content = f.read()
            print(f"   File size: {len(content)} bytes")
            
            # Check for common patterns
            content_str = content.decode('utf-8', errors='ignore').lower()
            print(f"   Contains 'part i': {' part i' in content_str}")
            print(f"   Contains 'part ii': {' part ii' in content_str}")
            print(f"   Contains 'item 1': {' item 1' in content_str}")
            
            # Show snippets where parts are mentioned
            if ' part i' in content_str:
                idx = content_str.find(' part i')
                snippet = content[max(0, idx-50):idx+100].decode('utf-8', errors='ignore')
                print(f"   Part I context: ...{snippet}...")
    
    # Process files
    start_time = datetime.now()
    total_sections = 0
    successful_files = 0
    
    cleaner = FilingCleaner(verbose=args.verbose)
    for file_path in tqdm(html_files, desc="Processing files"):
        sections = cleaner.process_file(file_path, output_dir)
        if sections > 0:
            successful_files += 1
        total_sections += sections
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n✅ Processing complete!")
    print(f"   Files processed: {len(html_files)}")
    print(f"   Files with output: {successful_files}")
    print(f"   Sections extracted: {total_sections}")
    print(f"   Time elapsed: {elapsed}")
    print(f"   Output files: {output_dir}/*.jsonl")
    
    if successful_files == 0:
        print(f"\n⚠️  WARNING: No files were successfully processed!")
        print(f"   Try running with --verbose --debug flags to see what's happening")
        print(f"   Make sure HTML files contain 'Part I' and 'Part II' sections")
        print(f"   File names should match pattern: YYYY-MM-DD_ticker-xxxx.html")


if __name__ == "__main__":
    main()