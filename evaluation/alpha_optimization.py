#!/usr/bin/env python
# alpha_optimization.py
"""
alpha_optimization.py — Alpha Parameter Optimization for Ticker-based Re-ranking
===============================================================================
가장 성능을 높일 수 있는 alpha 값을 찾는 스크립트

Features:
────────────────────────────────────────────
• Input
    - llm_labels.jsonl: Labeled query-chunk pairs with relevance annotations
      Format: {question, ticker, top_k, chunks: [{vec_id, rank, sim, yes, ticker}]}
• Alpha Adjustment
    - Boost similarity scores for chunks matching query ticker: sim + alpha
    - Re-rank chunks based on adjusted similarity scores
• Evaluation Metrics
    - Recall@k: Fraction of relevant chunks in top-k results
    - nDCG@k: Normalized Discounted Cumulative Gain at k
    - Calculate for k = 5, 10, 15, 20
• Optimization Process
    - Test multiple alpha values (default: 0.0 to 0.3, step 0.05)
    - Compute weighted average of all metrics
    - Select alpha with highest overall performance
• Output
    - alpha_optimization_results.txt: Detailed performance analysis
    - alpha_optimization_results.csv: Performance metrics table
    - Ticker distribution statistics

Usage:
    # Basic optimization with default parameters
    python alpha_optimization.py
    
    # Custom alpha range
    python alpha_optimization.py --alpha_min 0.0 --alpha_max 0.5 --alpha_step 0.01
    
    # Specify input/output files
    python alpha_optimization.py --input llm_labels.jsonl --output results.txt
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from typing import List, Dict, Tuple
from tqdm import tqdm

# Load labeled JSONL file
def load_labeled_data(jsonl_path: str) -> List[Dict]:
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Apply alpha weight to adjust similarity scores
def apply_alpha_adjustment(chunks: List[Dict], query_ticker: str, alpha: float) -> List[Dict]:
    adjusted_chunks = []
    for chunk in chunks:
        adjusted_chunk = chunk.copy()
        if chunk['ticker'] == query_ticker:
            adjusted_chunk['adjusted_sim'] = chunk['sim'] + alpha
        else:
            adjusted_chunk['adjusted_sim'] = chunk['sim']
        adjusted_chunks.append(adjusted_chunk)
    
    adjusted_chunks.sort(key=lambda x: x['adjusted_sim'], reverse=True)
    
    for i, chunk in enumerate(adjusted_chunks):
        chunk['adjusted_rank'] = i + 1
    
    return adjusted_chunks

# Calculate Recall@k metric
def calculate_recall_at_k(chunks: List[Dict], k: int) -> float:
    top_k_chunks = chunks[:k]
    relevant_in_top_k = sum(1 for chunk in top_k_chunks if chunk['yes'])
    total_relevant = sum(1 for chunk in chunks if chunk['yes'])
    
    if total_relevant == 0:
        return 0.0
    
    return relevant_in_top_k / total_relevant

# Calculate nDCG@k metric
def calculate_ndcg_at_k(chunks: List[Dict], k: int) -> float:
    y_true = [[1 if chunk['yes'] else 0 for chunk in chunks]]
    y_score = [[chunk['adjusted_sim'] for chunk in chunks]]
    
    return ndcg_score(y_true, y_score, k=k)

# Evaluate a single query with given alpha value
def evaluate_single_query(query_data: Dict, alpha: float) -> Dict[str, float]:
    chunks = query_data['chunks']
    query_ticker = query_data['ticker']
    
    adjusted_chunks = apply_alpha_adjustment(chunks, query_ticker, alpha)
    
    metrics = {
        'recall_at_5': calculate_recall_at_k(adjusted_chunks, 5),
        'recall_at_10': calculate_recall_at_k(adjusted_chunks, 10),
        'recall_at_15': calculate_recall_at_k(adjusted_chunks, 15),
        'recall_at_20': calculate_recall_at_k(adjusted_chunks, 20),
        'ndcg_at_5': calculate_ndcg_at_k(adjusted_chunks, 5),
        'ndcg_at_10': calculate_ndcg_at_k(adjusted_chunks, 10),
        'ndcg_at_15': calculate_ndcg_at_k(adjusted_chunks, 15),
        'ndcg_at_20': calculate_ndcg_at_k(adjusted_chunks, 20)
    }
    
    return metrics

# Evaluate performance of specific alpha value across all queries
def evaluate_alpha(labeled_data: List[Dict], alpha: float) -> Dict[str, float]:
    all_metrics = {
        'recall_at_5': [],
        'recall_at_10': [],
        'recall_at_15': [],
        'recall_at_20': [],
        'ndcg_at_5': [],
        'ndcg_at_10': [],
        'ndcg_at_15': [],
        'ndcg_at_20': []
    }
    
    for query_data in tqdm(labeled_data, desc=f"Evaluating alpha={alpha}", leave=False):
        metrics = evaluate_single_query(query_data, alpha)
        
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    avg_metrics = {
        key: np.mean(values) for key, values in all_metrics.items()
    }
    
    return avg_metrics

# Find optimal alpha value through grid search
def find_optimal_alpha(jsonl_path: str, alpha_values: List[float] = None) -> Tuple[float, pd.DataFrame]:
    if alpha_values is None:
        alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("Loading labeled data...")
    labeled_data = load_labeled_data(jsonl_path)
    print(f"Loaded {len(labeled_data)} queries")
    
    results = []
    
    print("Evaluating different alpha values...")
    for alpha in tqdm(alpha_values, desc="Alpha values"):
        metrics = evaluate_alpha(labeled_data, alpha)
        
        result = {'alpha': alpha}
        result.update(metrics)
        results.append(result)
        
        print(f"Alpha {alpha:.2f}: "
              f"Recall@5={metrics['recall_at_5']:.4f}, "
              f"Recall@10={metrics['recall_at_10']:.4f}, "
              f"Recall@15={metrics['recall_at_15']:.4f}, "
              f"Recall@20={metrics['recall_at_20']:.4f}, "
              f"nDCG@5={metrics['ndcg_at_5']:.4f}, "
              f"nDCG@10={metrics['ndcg_at_10']:.4f}, "
              f"nDCG@15={metrics['ndcg_at_15']:.4f}, "
              f"nDCG@20={metrics['ndcg_at_20']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    normalized_df = results_df.copy()
    for col in ['recall_at_5', 'recall_at_10', 'recall_at_15', 'recall_at_20', 'ndcg_at_5', 'ndcg_at_10', 'ndcg_at_15', 'ndcg_at_20']:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        if max_val > min_val:
            normalized_df[f'{col}_norm'] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'{col}_norm'] = 1.0
    
    normalized_df['weighted_score'] = (
        normalized_df['recall_at_5_norm'] * 0.125 +
        normalized_df['recall_at_10_norm'] * 0.125 +
        normalized_df['recall_at_15_norm'] * 0.125 +
        normalized_df['recall_at_20_norm'] * 0.125 +
        normalized_df['ndcg_at_5_norm'] * 0.125 +
        normalized_df['ndcg_at_10_norm'] * 0.125 +
        normalized_df['ndcg_at_15_norm'] * 0.125 +
        normalized_df['ndcg_at_20_norm'] * 0.125
    )
    
    best_idx = normalized_df['weighted_score'].idxmax()
    optimal_alpha = results_df.loc[best_idx, 'alpha']
    
    return optimal_alpha, results_df

# Analyze distribution of same/different ticker chunks
def analyze_ticker_distribution(labeled_data: List[Dict]) -> pd.DataFrame:
    ticker_stats = {}
    
    for query_data in labeled_data:
        query_ticker = query_data['ticker']
        chunks = query_data['chunks']
        
        if query_ticker not in ticker_stats:
            ticker_stats[query_ticker] = {
                'query_count': 0,
                'same_ticker_chunks': 0,
                'different_ticker_chunks': 0,
                'same_ticker_relevant': 0,
                'different_ticker_relevant': 0
            }
        
        ticker_stats[query_ticker]['query_count'] += 1
        
        for chunk in chunks:
            if chunk['ticker'] == query_ticker:
                ticker_stats[query_ticker]['same_ticker_chunks'] += 1
                if chunk['yes']:
                    ticker_stats[query_ticker]['same_ticker_relevant'] += 1
            else:
                ticker_stats[query_ticker]['different_ticker_chunks'] += 1
                if chunk['yes']:
                    ticker_stats[query_ticker]['different_ticker_relevant'] += 1
    
    stats_df = pd.DataFrame(ticker_stats).T
    
    stats_df['same_ticker_relevance_rate'] = (
        stats_df['same_ticker_relevant'] / stats_df['same_ticker_chunks']
    ).fillna(0)
    
    stats_df['different_ticker_relevance_rate'] = (
        stats_df['different_ticker_relevant'] / stats_df['different_ticker_chunks']
    ).fillna(0)
    
    return stats_df

# Save detailed results to text file
def save_detailed_results(results_df: pd.DataFrame, ticker_stats: pd.DataFrame, 
                         optimal_alpha: float, output_path: str = "alpha_optimization_results.txt"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Alpha Optimization Results ===\n\n")
        
        f.write(f"Optimal Alpha: {optimal_alpha:.2f}\n\n")
        
        f.write("Performance by Alpha Values:\n")
        f.write(results_df.to_string(index=False, float_format='%.4f'))
        f.write("\n\n")
        
        f.write("Ticker Distribution Analysis:\n")
        f.write(ticker_stats.to_string(float_format='%.4f'))
        f.write("\n\n")
        
        f.write("Best Performance Details:\n")
        best_row = results_df[results_df['alpha'] == optimal_alpha].iloc[0]
        for metric in ['recall_at_5', 'recall_at_10', 'recall_at_15', 'recall_at_20', 'ndcg_at_5', 'ndcg_at_10', 'ndcg_at_15', 'ndcg_at_20']:
            f.write(f"{metric}: {best_row[metric]:.4f}\n")

# Main execution function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find optimal alpha for ticker-based re-ranking")
    parser.add_argument("--input", default="llm_labels.jsonl", 
                       help="Input JSONL file with labeled data")
    parser.add_argument("--output", default="alpha_optimization_results.txt",
                       help="Output file for detailed results")
    parser.add_argument("--alpha_min", type=float, default=0.0,
                       help="Minimum alpha value")
    parser.add_argument("--alpha_max", type=float, default=0.3,
                       help="Maximum alpha value")
    parser.add_argument("--alpha_step", type=float, default=0.05,
                       help="Alpha step size")
    
    args = parser.parse_args()
    
    alpha_values = np.arange(args.alpha_min, args.alpha_max + args.alpha_step/2, args.alpha_step).tolist()
    
    print(f"Testing alpha values: {alpha_values}")
    
    optimal_alpha, results_df = find_optimal_alpha(args.input, alpha_values)
    
    labeled_data = load_labeled_data(args.input)
    ticker_stats = analyze_ticker_distribution(labeled_data)
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Optimal Alpha: {optimal_alpha:.2f}")
    print("\nPerformance Summary:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    save_detailed_results(results_df, ticker_stats, optimal_alpha, args.output)
    print(f"\nDetailed results saved to: {args.output}")
    
    csv_path = args.output.replace('.txt', '.csv')
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Results DataFrame saved to: {csv_path}")

if __name__ == "__main__":
    main()