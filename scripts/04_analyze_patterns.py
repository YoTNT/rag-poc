"""
Analyze whether similar-context accuracy predicts query accuracy.

This is Step 4 of the RAG POC pipeline — the decision-making script.

For every prediction in the corpus:
  1. Find its N most similar historical predictions
  2. Compute the accuracy of that "similar context"
  3. Record (query_wasCorrect, similar_context_accuracy)

Then bucket queries by their similar-context accuracy:
  - High-context (similar N > HIGH_THRESHOLD accurate)
  - Low-context  (similar N < LOW_THRESHOLD accurate)

If high-context queries have meaningfully higher accuracy than
low-context queries, RAG has predictive signal — proceed to Phase 1.

Usage:
    python scripts/04_analyze_patterns.py
    python scripts/04_analyze_patterns.py --k 10   # use 10 similar instead of 20

Expected runtime: ~10 seconds
Output: console report + data/pattern_analysis.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = Path(__file__).parent.parent / 'data' / 'embeddings.pkl'
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'pattern_analysis.csv'

# Bucket thresholds for "high" vs "low" context confidence.
# Baseline accuracy is ~44%, so >60% and <40% are meaningfully off baseline.
HIGH_CONTEXT_THRESHOLD = 0.60
LOW_CONTEXT_THRESHOLD = 0.40


def compute_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute full N×N cosine similarity matrix once.

    With 710 predictions this is a 710×710 matrix = ~2 MB, trivial.
    Much faster than looping and computing 710 individual 1×N queries.
    """
    print('Computing 710×710 similarity matrix (one-time, ~1 sec)...')
    return cosine_similarity(matrix)


def analyze(
    data: list[dict],
    similarity_matrix: np.ndarray,
    k: int = 20,
) -> pd.DataFrame:
    """
    For every prediction, find its top-K similar predictions (excluding
    itself), and compute the accuracy of that context.

    Returns a DataFrame with one row per query:
      - query_id
      - query_symbol
      - query_correct (bool)
      - query_prompt_version
      - similar_accuracy (float — proportion of top-K that were correct)
      - similar_valid_count (int — how many of top-K had wasCorrect set)
    """
    n = len(data)
    rows = []

    for i in range(n):
        query = data[i]

        # Skip queries with unknown wasCorrect (defensive — should be all True)
        if query['metadata'].get('wasCorrect') is None:
            continue

        # Top-K similar (excluding self)
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # exclude self by pushing similarity below all others
        top_indices = np.argsort(sims)[::-1][:k]

        # Compute accuracy of the top-K
        similar_correct_count = 0
        similar_valid_count = 0
        for idx in top_indices:
            was_correct = data[idx]['metadata'].get('wasCorrect')
            if was_correct is not None:
                similar_valid_count += 1
                if was_correct:
                    similar_correct_count += 1

        # Skip if no valid similar predictions (edge case — shouldn't happen)
        if similar_valid_count == 0:
            continue

        similar_accuracy = similar_correct_count / similar_valid_count

        rows.append({
            'query_id': query['id'],
            'query_symbol': query['metadata']['symbol'],
            'query_correct': query['metadata']['wasCorrect'],
            'query_prompt_version': query['metadata'].get('promptVersion', 'unknown'),
            'similar_accuracy': similar_accuracy,
            'similar_valid_count': similar_valid_count,
        })

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, k: int) -> None:
    """Print the key findings — this is the POC decision report output."""
    print()
    print('=' * 70)
    print(f'RAG Signal Analysis (using top-{k} similar as context)')
    print('=' * 70)
    print()

    total = len(df)
    baseline_accuracy = df['query_correct'].mean()

    print(f'Total queries analyzed: {total}')
    print(f'Baseline accuracy (all queries): {baseline_accuracy:.1%}')
    print()

    # Bucket queries by their similar-context accuracy
    high_ctx = df[df['similar_accuracy'] > HIGH_CONTEXT_THRESHOLD]
    low_ctx = df[df['similar_accuracy'] < LOW_CONTEXT_THRESHOLD]
    mid_ctx = df[
        (df['similar_accuracy'] >= LOW_CONTEXT_THRESHOLD)
        & (df['similar_accuracy'] <= HIGH_CONTEXT_THRESHOLD)
    ]

    print(f'--- Bucket analysis ---')
    print(f'Bucket thresholds: high > {HIGH_CONTEXT_THRESHOLD:.0%}, low < {LOW_CONTEXT_THRESHOLD:.0%}')
    print()
    print(f'{"Bucket":<20} {"Queries":<10} {"Query Acc":<12} {"vs Baseline"}')
    print('-' * 60)

    for name, group in [('high-context', high_ctx), ('mid-context', mid_ctx), ('low-context', low_ctx)]:
        if len(group) == 0:
            print(f'{name:<20} {0:<10} {"—":<12} {"—"}')
            continue
        acc = group['query_correct'].mean()
        delta = acc - baseline_accuracy
        delta_str = f'{delta:+.1%}' if delta != 0 else '±0.0%'
        print(f'{name:<20} {len(group):<10} {acc:.1%}     {delta_str}')

    print()

    # The key metric: delta between high-context and low-context
    if len(high_ctx) > 0 and len(low_ctx) > 0:
        delta_pp = (high_ctx['query_correct'].mean() - low_ctx['query_correct'].mean()) * 100
        print(f'*** High vs Low context delta: {delta_pp:+.1f} percentage points ***')
        print()

        if delta_pp >= 10:
            print('✅ RAG has meaningful predictive signal (delta ≥ 10pp)')
            print('   Recommendation: PROCEED to Phase 1 (production RAG service)')
        elif delta_pp >= 5:
            print('⚠️  RAG has weak-to-moderate signal (delta 5-10pp)')
            print('   Recommendation: More data needed, or try different embedding strategy')
        else:
            print('❌ RAG has no meaningful signal (delta < 5pp)')
            print('   Recommendation: Do not pursue Phase 1 with current approach')
    else:
        print('⚠️  Not enough queries in high/low buckets to compare')

    print()
    print('--- Per-prompt-version breakdown ---')
    per_version = df.groupby('query_prompt_version').agg(
        count=('query_correct', 'size'),
        baseline_acc=('query_correct', 'mean'),
    ).sort_values('count', ascending=False)
    print(per_version.to_string())
    print()

    print(f'Full results written to: {OUTPUT_PATH}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze RAG predictive signal.')
    parser.add_argument('--k', type=int, default=20, help='Number of similar predictions per query (default 20)')
    args = parser.parse_args()

    print(f'Loading {EMBEDDINGS_PATH}...')
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)

    matrix = np.array([item['embedding'] for item in data], dtype=np.float32)
    print(f'Loaded {len(data)} embeddings, matrix shape {matrix.shape}')

    similarity_matrix = compute_similarity_matrix(matrix)

    print(f'Analyzing patterns (k={args.k})...')
    df = analyze(data, similarity_matrix, k=args.k)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print_report(df, k=args.k)


if __name__ == '__main__':
    main()