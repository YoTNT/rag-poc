"""
Compare embedding strategies to find which gives strongest RAG signal.

This is Step 5 of the RAG POC pipeline — validates that the "structured"
embedding chosen in step 2 is actually the best option.

Compared strategies:
  1. reasoning_only    — embed just the reasoning text
  2. indicators_only   — embed just the numeric indicators (as text)
  3. structured        — symbol + direction + indicators + reasoning (current)

For each strategy:
  - Generate fresh embeddings for all 710 predictions
  - Run the same pattern analysis as step 4
  - Report high-vs-low context delta

Winner = strategy with largest delta (strongest predictive signal).

Usage:
    python scripts/05_compare_strategies.py

Expected runtime: ~5 minutes (3 × embedding generation)
Expected cost: ~$0.01 (3 × ~$0.003)
Output: console report + data/strategy_comparison.csv
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

INPUT_PATH = Path(__file__).parent.parent / 'data' / 'predictions.json'
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'strategy_comparison.csv'

EMBEDDING_MODEL = 'text-embedding-3-small'
BATCH_SIZE = 100
K_SIMILAR = 20

HIGH_CONTEXT_THRESHOLD = 0.60
LOW_CONTEXT_THRESHOLD = 0.40

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# ─── Embedding input strategies ─────────────────────────────

def strategy_reasoning_only(prediction: dict) -> str:
    """Just the reasoning text — nothing else."""
    return prediction.get('reasoning', '')


def strategy_indicators_only(prediction: dict) -> str:
    """
    Only numeric indicators, described in short text form.
    Falls back to symbol+direction when snapshot is missing so we don't
    end up with empty strings.
    """
    parts = [f"Symbol: {prediction['symbol']}", f"Direction: {prediction['direction']}"]
    indicators = prediction.get('indicatorsSnapshot')
    if indicators:
        if indicators.get('rsi14') is not None:
            parts.append(f"RSI {indicators['rsi14']:.1f}")
        if indicators.get('macd') is not None:
            parts.append(f"MACD {indicators['macd']:.2f}")
        if indicators.get('volumeRatio') is not None:
            parts.append(f"Volume {indicators['volumeRatio']:.2f}x")
        if indicators.get('sma20') is not None:
            parts.append(f"SMA20 {indicators['sma20']:.2f}")
        if indicators.get('sma50') is not None:
            parts.append(f"SMA50 {indicators['sma50']:.2f}")
    return ' | '.join(parts)


def strategy_structured(prediction: dict) -> str:
    """The same structured format used in script 02 (baseline)."""
    parts = [
        f"Symbol: {prediction['symbol']}",
        f"Date: {prediction['predictedFor']}",
        f"Direction: {prediction['direction']}",
        f"Confidence: {prediction.get('confidence', 'N/A')}",
        f"Prompt: {prediction.get('promptVersion', 'unknown')}",
    ]
    indicators = prediction.get('indicatorsSnapshot')
    if indicators:
        ind_parts = []
        if indicators.get('rsi14') is not None:
            ind_parts.append(f"RSI {indicators['rsi14']:.1f}")
        if indicators.get('macd') is not None:
            ind_parts.append(f"MACD {indicators['macd']:.2f}")
        if indicators.get('volumeRatio') is not None:
            ind_parts.append(f"Volume {indicators['volumeRatio']:.2f}x")
        if ind_parts:
            parts.append(f"Indicators: {', '.join(ind_parts)}")
    if prediction.get('reasoning'):
        parts.append(f"Reasoning: {prediction['reasoning']}")
    return '\n'.join(parts)


STRATEGIES = {
    'reasoning_only': strategy_reasoning_only,
    'indicators_only': strategy_indicators_only,
    'structured': strategy_structured,
}


# ─── Embedding + analysis helpers ───────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Same batching pattern as script 02."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def generate_embeddings_for_strategy(
    predictions: list[dict],
    strategy_name: str,
    build_input,
) -> list[dict]:
    """Run one strategy end-to-end: build inputs → batch embed → return list."""
    print(f'\n--- Strategy: {strategy_name} ---')
    print(f'  Building embedding inputs...')

    prepared = []
    for p in predictions:
        if not p.get('reasoning'):
            continue
        prepared.append({
            'id': p['id'],
            'metadata': {
                'wasCorrect': p.get('wasCorrect'),
            },
            'input_text': build_input(p),
        })

    all_embeddings = []
    total_batches = (len(prepared) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[batch_idx : batch_idx + BATCH_SIZE]
        texts = [item['input_text'] for item in batch]
        current = batch_idx // BATCH_SIZE + 1
        print(f'  Batch {current}/{total_batches} ({len(texts)} predictions)...')
        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)

    return [
        {
            'id': prepared[i]['id'],
            'metadata': prepared[i]['metadata'],
            'embedding': all_embeddings[i],
        }
        for i in range(len(prepared))
    ]


def analyze_signal(data: list[dict]) -> dict:
    """
    Same analysis as script 04, but returns numbers rather than printing.
    Returns dict with baseline, high_ctx_acc, low_ctx_acc, delta_pp, sample counts.
    """
    matrix = np.array([item['embedding'] for item in data], dtype=np.float32)
    similarity_matrix = cosine_similarity(matrix)

    rows = []
    for i, query in enumerate(data):
        if query['metadata'].get('wasCorrect') is None:
            continue
        sims = similarity_matrix[i].copy()
        sims[i] = -1
        top_indices = np.argsort(sims)[::-1][:K_SIMILAR]

        similar_correct = 0
        similar_valid = 0
        for idx in top_indices:
            was_correct = data[idx]['metadata'].get('wasCorrect')
            if was_correct is not None:
                similar_valid += 1
                if was_correct:
                    similar_correct += 1

        if similar_valid == 0:
            continue

        rows.append({
            'query_correct': query['metadata']['wasCorrect'],
            'similar_accuracy': similar_correct / similar_valid,
        })

    df = pd.DataFrame(rows)
    baseline = df['query_correct'].mean()
    high_ctx = df[df['similar_accuracy'] > HIGH_CONTEXT_THRESHOLD]
    low_ctx = df[df['similar_accuracy'] < LOW_CONTEXT_THRESHOLD]

    high_acc = high_ctx['query_correct'].mean() if len(high_ctx) > 0 else None
    low_acc = low_ctx['query_correct'].mean() if len(low_ctx) > 0 else None

    delta_pp = None
    if high_acc is not None and low_acc is not None:
        delta_pp = (high_acc - low_acc) * 100

    return {
        'baseline_acc': baseline,
        'high_ctx_count': len(high_ctx),
        'high_ctx_acc': high_acc,
        'low_ctx_count': len(low_ctx),
        'low_ctx_acc': low_acc,
        'delta_pp': delta_pp,
    }


# ─── Main ───────────────────────────────────────────────────

def main() -> None:
    print(f'Loading {INPUT_PATH}...')
    with open(INPUT_PATH) as f:
        predictions = json.load(f)
    print(f'Loaded {len(predictions)} predictions')

    results = []
    for strategy_name, build_input in STRATEGIES.items():
        data = generate_embeddings_for_strategy(predictions, strategy_name, build_input)
        signal = analyze_signal(data)
        results.append({
            'strategy': strategy_name,
            **signal,
        })

    # Print comparison table
    print()
    print('=' * 90)
    print('Strategy Comparison')
    print('=' * 90)
    print()
    print(f'{"Strategy":<20} {"Baseline":<12} {"High Ctx":<20} {"Low Ctx":<20} {"Delta (pp)"}')
    print('-' * 90)

    for r in results:
        high_str = f"{r['high_ctx_acc']:.1%} (n={r['high_ctx_count']})" if r['high_ctx_acc'] is not None else '—'
        low_str = f"{r['low_ctx_acc']:.1%} (n={r['low_ctx_count']})" if r['low_ctx_acc'] is not None else '—'
        delta_str = f"{r['delta_pp']:+.1f}" if r['delta_pp'] is not None else '—'
        print(f"{r['strategy']:<20} {r['baseline_acc']:.1%}{'':<7} {high_str:<20} {low_str:<20} {delta_str}")

    print()

    # Winner
    ranked = sorted(
        (r for r in results if r['delta_pp'] is not None),
        key=lambda r: r['delta_pp'],
        reverse=True,
    )
    if ranked:
        winner = ranked[0]
        print(f'*** Winner: {winner["strategy"]} with delta {winner["delta_pp"]:+.1f} pp ***')
        print()
        print(f'Recommendation: Use "{winner["strategy"]}" for Phase 1.')

    # Save CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f'\nFull results written to: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()