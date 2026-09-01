"""
Search for predictions similar to a given query prediction.

This is Step 3 of the RAG POC pipeline — the interactive tool used
to eyeball retrieval quality.

Loads embeddings from data/embeddings.pkl, computes cosine similarity
between the query and every other prediction, prints the top N.

Usage:
    # Basic: top 5 similar to a given prediction
    python scripts/03_search_similar.py --query-id a88ea78e-b368-...

    # Top 10, only AAPL predictions
    python scripts/03_search_similar.py --query-id <id> --n 10 --symbol AAPL

    # Only show predictions that were wrong (useful for failure-mode analysis)
    python scripts/03_search_similar.py --query-id <id> --only-wrong

    # Pick a random query if you don't have an id handy
    python scripts/03_search_similar.py --random

Expected runtime: <1 second (in-memory computation)
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = Path(__file__).parent.parent / 'data' / 'embeddings.pkl'


def load_embeddings() -> tuple[list[dict], np.ndarray, list[str]]:
    """
    Load embeddings.pkl and return:
      - full data list (for metadata lookup)
      - numpy matrix of embeddings, shape (N, 1536)
      - id-to-index mapping (list of ids in matrix order)

    Building the matrix once at load time makes each search O(N) instead
    of re-materializing the matrix per query.
    """
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)

    ids = [item['id'] for item in data]
    matrix = np.array([item['embedding'] for item in data], dtype=np.float32)

    return data, matrix, ids


def find_similar(
    query_id: str,
    data: list[dict],
    matrix: np.ndarray,
    ids: list[str],
    n: int = 5,
    filter_symbol: str | None = None,
    only_wrong: bool = False,
) -> list[tuple[dict, float]]:
    """
    Find top-N predictions similar to query_id.

    Args:
        query_id: id of the prediction to search from
        data: full prediction data list (in matrix order)
        matrix: (N, 1536) embedding matrix
        ids: list of prediction ids matching matrix rows
        n: how many results to return
        filter_symbol: only include predictions for this symbol
        only_wrong: only include predictions where wasCorrect == False

    Returns:
        List of (prediction_dict, similarity_score) tuples, sorted by
        similarity descending. Query itself is excluded.
    """
    if query_id not in ids:
        raise ValueError(f'Query id {query_id} not found in embeddings')

    query_idx = ids.index(query_id)
    query_vec = matrix[query_idx].reshape(1, -1)

    # Cosine similarity between query and every other row.
    # sklearn returns shape (1, N); flatten to (N,).
    similarities = cosine_similarity(query_vec, matrix)[0]

    # Sort indices by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]

    results: list[tuple[dict, float]] = []
    for idx in sorted_indices:
        candidate = data[idx]

        # Skip the query itself
        if candidate['id'] == query_id:
            continue

        # Apply filters
        if filter_symbol and candidate['metadata']['symbol'] != filter_symbol.upper():
            continue
        if only_wrong and candidate['metadata'].get('wasCorrect') is not False:
            continue

        results.append((candidate, float(similarities[idx])))

        if len(results) >= n:
            break

    return results


def format_prediction_summary(prediction: dict, similarity: float | None = None) -> str:
    """Format one prediction as a compact multi-line summary for the terminal."""
    m = prediction['metadata']
    correct_marker = {
        True: '✓',
        False: '✗',
        None: '?',
    }.get(m.get('wasCorrect'), '?')

    header = f"[{correct_marker}] {m['symbol']:<5} {m['predictedFor']}  {m['direction']:<8} conf={m.get('confidence', 'N/A')}  {m.get('promptVersion', 'unknown')}"
    if similarity is not None:
        header = f"  (sim={similarity:.3f})  " + header
    else:
        header = "  QUERY:     " + header

    reasoning = prediction['input_text'].split('Reasoning: ', 1)
    reasoning_snippet = reasoning[1][:180] if len(reasoning) == 2 else '(no reasoning)'
    return f'{header}\n            {reasoning_snippet}...'


def main() -> None:
    parser = argparse.ArgumentParser(description='Find predictions similar to a query.')
    parser.add_argument('--query-id', help='Prediction id to search from')
    parser.add_argument('--random', action='store_true', help='Pick a random query id')
    parser.add_argument('--n', type=int, default=5, help='Number of similar predictions to return (default 5)')
    parser.add_argument('--symbol', help='Filter results to this symbol only')
    parser.add_argument('--only-wrong', action='store_true', help='Only include predictions where wasCorrect == False')
    args = parser.parse_args()

    if not args.query_id and not args.random:
        parser.error('Must provide either --query-id or --random')

    data, matrix, ids = load_embeddings()
    print(f'Loaded {len(data)} embeddings\n')

    query_id = args.query_id or random.choice(ids)
    query = next(d for d in data if d['id'] == query_id)

    print('=' * 90)
    print(format_prediction_summary(query))
    print('=' * 90)
    print()

    filter_desc = []
    if args.symbol:
        filter_desc.append(f'symbol={args.symbol.upper()}')
    if args.only_wrong:
        filter_desc.append('only_wrong=True')
    filter_note = f' ({", ".join(filter_desc)})' if filter_desc else ''

    print(f'Top {args.n} similar predictions{filter_note}:\n')

    results = find_similar(
        query_id=query_id,
        data=data,
        matrix=matrix,
        ids=ids,
        n=args.n,
        filter_symbol=args.symbol,
        only_wrong=args.only_wrong,
    )

    if not results:
        print('  (no results matching filters)')
        return

    for i, (candidate, similarity) in enumerate(results, 1):
        print(f'  {i}.', end='')
        print(format_prediction_summary(candidate, similarity=similarity))
        print()


if __name__ == '__main__':
    main()