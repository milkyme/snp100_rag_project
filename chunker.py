#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunker.py - SEC 10-K Document Chunker
=======================================
통합된 10-K 문서 청킹 및 중복 제거 스크립트

Features:
- Variable chunk size based on section importance (256-1024 tokens)
- Configurable overlap between chunks (0-100%)
- Automatic deduplication within files
- Smart sentence splitting with NLTK

Usage:
    python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks
    python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks --overlap 20
    python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks --verbose
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Generator
from datetime import datetime

import tiktoken
import nltk
from tqdm import tqdm

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)

# Document chunker with configurable overlap and deduplication
class DocumentChunker:
    
    # Token limits
    MIN_CHUNK_SIZE = 256
    MAX_CHUNK_SIZE = 1024
    
    def __init__(self, overlap_percent: int = 0, verbose: bool = False):
        """
        Initialize chunker with overlap percentage
        
        Args:
            overlap_percent: Percentage of overlap between chunks (0-100)
            verbose: Enable detailed logging
        """
        if not 0 <= overlap_percent <= 100:
            raise ValueError("Overlap percentage must be between 0 and 100")
        
        self.overlap_percent = overlap_percent
        self.verbose = verbose
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        if self.verbose:
            print(f"Initialized chunker with {overlap_percent}% overlap")
    
    # Calculate token length of text using tiktoken
    def token_length(self, text: str) -> int:
        return len(self.encoder.encode_ordinary(text))
    
    # Split text into sentences, handling edge cases
    # Long sentences exceeding MAX_CHUNK_SIZE are forcibly split
    def split_sentences(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        
        # Handle overly long sentences
        result = []
        for sent in sentences:
            if self.token_length(sent) > self.MAX_CHUNK_SIZE:
                # Force split long sentences at token boundaries
                tokens = self.encoder.encode_ordinary(sent)
                for i in range(0, len(tokens), self.MAX_CHUNK_SIZE):
                    chunk_tokens = tokens[i:i + self.MAX_CHUNK_SIZE]
                    chunk_text = self.encoder.decode(chunk_tokens)
                    result.append(chunk_text)
            else:
                result.append(sent)
        
        return result
    
    # Calculate target chunk size based on section weight
    # Higher weight → smaller chunks (more important sections)
    def calculate_target_size(self, weight: float) -> int:
        return int(self.MIN_CHUNK_SIZE + 
                  (self.MAX_CHUNK_SIZE - self.MIN_CHUNK_SIZE) * (1 - weight))
    
    # Create chunks from text with configurable overlap
    # Yields tuples of (chunk_text, token_count)
    def create_chunks(self, text: str, weight: float) -> Generator[Tuple[str, int], None, None]:
        sentences = self.split_sentences(text)
        if not sentences:
            return
        
        token_lengths = [self.token_length(sent) for sent in sentences]
        target_size = self.calculate_target_size(weight)
        
        # Calculate overlap in sentences
        if self.overlap_percent > 0 and len(sentences) > 1:
            # Estimate sentences per chunk
            avg_sent_tokens = sum(token_lengths) / len(token_lengths)
            sents_per_chunk = max(1, int(target_size / avg_sent_tokens))
            overlap_sents = max(1, int(sents_per_chunk * self.overlap_percent / 100))
        else:
            overlap_sents = 0
        
        i = 0
        n = len(sentences)
        
        while i < n:
            # Build chunk starting from position i
            chunk_start = i
            current_tokens = 0
            
            # Add sentences until target size is reached
            while i < n and current_tokens + token_lengths[i] <= target_size:
                current_tokens += token_lengths[i]
                i += 1
            
            # Handle case where single sentence exceeds target
            if chunk_start == i and i < n:
                current_tokens = token_lengths[i]
                i += 1
            
            # Create chunk text
            chunk_sentences = sentences[chunk_start:i]
            chunk_text = " ".join(chunk_sentences)
            
            yield chunk_text, current_tokens
            
            # Apply overlap by moving back
            if overlap_sents > 0 and i < n:
                i = max(chunk_start + 1, i - overlap_sents)
    
    # Compute SHA256 hash of text for deduplication
    def compute_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()
    
    # Process sections into chunks with deduplication
    # Returns a list of chunk records with proper indexing
    def process_sections(self, sections: List[Dict]) -> List[Dict]:
        all_chunks = []
        seen_hashes = set()
        
        # Group by (part, section_item) for proper indexing
        section_groups = {}
        for section in sections:
            key = (section["part"], section["section_item"])
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(section)
        
        # Process each section group
        for (part, section_item), group_sections in section_groups.items():
            group_chunks = []
            
            for section in group_sections:
                weight = section.get("section_weight", 0.5)
                text = section.get("text", "")
                
                if not text.strip():
                    continue
                
                # Create chunks for this section
                for chunk_text, token_count in self.create_chunks(text, weight):
                    # Check for duplicates
                    chunk_hash = self.compute_text_hash(chunk_text)
                    if chunk_hash in seen_hashes:
                        if self.verbose:
                            print(f"  Skipping duplicate chunk in {section_item}")
                        continue
                    
                    seen_hashes.add(chunk_hash)
                    
                    # Create chunk record
                    chunk_record = {
                        **section,  # Inherit all section metadata
                        "chunk_size": token_count,
                        "text": chunk_text,
                        "chunk_hash": chunk_hash[:8]  # Store first 8 chars for reference
                    }
                    group_chunks.append(chunk_record)
            
            # Assign chunk indices within the group
            for idx, chunk in enumerate(group_chunks, 1):
                chunk["chunk_index"] = idx
                chunk["total_chunks_in_section"] = len(group_chunks)
            
            all_chunks.extend(group_chunks)
        
        # Sort by part → section → chunk index
        all_chunks.sort(key=lambda x: (
            x["part"],
            x["section_item"],
            x["chunk_index"]
        ))
        
        return all_chunks
    
    # Process a single JSONL file
    # Returns number of chunks created
    def process_file(self, input_path: Path, output_dir: Path) -> int:
        try:
            # Read input file
            with input_path.open('r', encoding='utf-8') as f:
                sections = [json.loads(line) for line in f if line.strip()]
            
            if not sections:
                if self.verbose:
                    print(f"[SKIP] Empty file: {input_path.name}")
                return 0
            
            if self.verbose:
                print(f"\n[PROCESSING] {input_path.name}")
                print(f"  Sections: {len(sections)}")
            
            # Process sections into chunks
            chunks = self.process_sections(sections)
            
            if not chunks:
                if self.verbose:
                    print(f"  No chunks generated")
                return 0
            
            # Prepare output file
            output_path = output_dir / f"{input_path.stem}_chunks.jsonl"
            
            # Write chunks
            with output_path.open('w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
            if self.verbose:
                unique_sections = set((c["part"], c["section_item"]) for c in chunks)
                print(f"  Chunks created: {len(chunks)}")
                print(f"  Unique sections: {len(unique_sections)}")
                print(f"  Output: {output_path.name}")
            
            return len(chunks)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {input_path.name}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Chunk SEC 10-K documents with configurable overlap and deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Basic usage (no overlap)
                python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks
                
                # With 20% overlap between chunks
                python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks --overlap 20
                
                # With verbose output
                python chunker.py --in_dir ./10k_cleaned --out_dir ./10k_chunks --verbose

            Overlap explanation:
                - 0% overlap: No sentence appears in multiple chunks (default)
                - 50% overlap: ~50% of sentences in each chunk also appear in adjacent chunks
                - 100% overlap: Maximum overlap (sliding window approach)
            """
    )
    
    parser.add_argument("--in_dir", required=True, 
                       help="Directory containing cleaned JSONL files")
    parser.add_argument("--out_dir", required=True, 
                       help="Output directory for chunked JSONL files")
    parser.add_argument("--overlap", type=int, default=0,
                       help="Overlap percentage between chunks (0-100, default: 0)")
    parser.add_argument("--limit", type=int,
                       help="Process only first N files (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.overlap <= 100:
        parser.error("Overlap must be between 0 and 100")
    
    # Setup paths
    input_dir = Path(args.in_dir)
    output_dir = Path(args.out_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    input_files = list(input_dir.glob("*.jsonl"))
    # Exclude chunk files if any exist in input dir
    input_files = [f for f in input_files if "_chunks" not in f.name]
    
    if args.limit:
        input_files = input_files[:args.limit]
    
    if not input_files:
        print(f"❌ No JSONL files found in {input_dir}")
        return
    
    print(f"📁 Found {len(input_files)} files to process")
    print(f"📝 Output directory: {output_dir}")
    print(f"🔄 Overlap setting: {args.overlap}%")
    
    # Process files
    start_time = datetime.now()
    total_chunks = 0
    successful_files = 0
    
    # Initialize chunker
    chunker = DocumentChunker(overlap_percent=args.overlap, verbose=args.verbose)
    
    # Process files sequentially
    for file_path in tqdm(input_files, desc="Processing files"):
        chunks_created = chunker.process_file(file_path, output_dir)
        if chunks_created > 0:
            successful_files += 1
            total_chunks += chunks_created
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n✅ Chunking complete!")
    print(f"   Files processed: {len(input_files)}")
    print(f"   Files with output: {successful_files}")
    print(f"   Total chunks created: {total_chunks}")
    print(f"   Average chunks per file: {total_chunks / successful_files:.1f}" 
          if successful_files > 0 else "")
    print(f"   Time elapsed: {elapsed}")
    print(f"   Output files: {output_dir}/*_chunks.jsonl")


if __name__ == "__main__":
    main()