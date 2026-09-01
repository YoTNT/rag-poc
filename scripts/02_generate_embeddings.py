"""
Generate embeddings for all verified predictions via OpenAI.

This is Step 2 of the RAG POC pipeline.

- Reads data/predictions.json (from step 1)
- Builds a structured text representation for each prediction
  (symbol + direction + indicators + reasoning) — see build_embedding_input()
- Batches the OpenAI embedding API calls (100 predictions per request)
- Saves data/embeddings.pkl containing embedding + metadata + input_text

Usage:
    python scripts/02_generate_embeddings.py

Expected runtime: ~1 minute
Expected cost: <$0.01 (text-embedding-3-small at $0.02 per 1M tokens)
Output: data/embeddings.pkl (~4 MB)
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_PATH = Path(__file__).parent.parent / 'data' / 'predictions.json'
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'embeddings.pkl'

EMBEDDING_MODEL = 'text-embedding-3-small'
BATCH_SIZE = 100   # OpenAI accepts up to 2048 inputs per request; 100 is safe

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def build_embedding_input(prediction: dict) -> str:
    """
    Construct a structured text representation for embedding.

    Combines symbol + date + direction + key indicators + reasoning
    into a single string. This is embedded as one vector.

    Rationale for including indicators alongside reasoning:
    - Reasoning alone captures Claude's thought process but is fuzzy
      on exact numbers (RSI 47 vs RSI 48 read the same)
    - Indicators give numeric grounding
    - Combining both = semantic + quantitative similarity
    """
    parts = [
        f"Symbol: {prediction['symbol']}",
        f"Date: {prediction['predictedFor']}",
        f"Direction: {prediction['direction']}",
        f"Confidence: {prediction.get('confidence', 'N/A')}",
        f"Prompt: {prediction.get('promptVersion', 'unknown')}",
    ]

    # Add key indicators if snapshot is present.
    # ~14% of predictions (early v1) lack indicatorsSnapshot — that's fine.
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

    # Reasoning goes last (most content-rich part).
    if prediction.get('reasoning'):
        parts.append(f"Reasoning: {prediction['reasoning']}")

    return '\n'.join(parts)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI embeddings API for a batch of texts.
    Returns list of embeddings in the same order as inputs.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main() -> None:
    print(f'Loading {INPUT_PATH}...')
    with open(INPUT_PATH) as f:
        predictions = json.load(f)

    print(f'Loaded {len(predictions)} predictions\n')

    # Build (metadata, embedding_input) pairs for every prediction with reasoning.
    # Skip predictions missing reasoning (shouldn't happen based on step 1 check
    # but this is defensive).
    prepared: list[dict[str, Any]] = []
    skipped = 0
    for p in predictions:
        if not p.get('reasoning'):
            skipped += 1
            continue
        prepared.append({
            'id': p['id'],
            'metadata': {
                'symbol': p['symbol'],
                'predictedFor': p['predictedFor'],
                'direction': p['direction'],
                'wasCorrect': p.get('wasCorrect'),
                'promptVersion': p.get('promptVersion'),
                'basePrice': p.get('basePrice'),
                'actualPrice': p.get('actualPrice'),
                'confidence': p.get('confidence'),
            },
            'input_text': build_embedding_input(p),
        })

    if skipped:
        print(f'⚠️  Skipped {skipped} predictions with no reasoning')

    print(f'Preparing to embed {len(prepared)} predictions in batches of {BATCH_SIZE}\n')

    # Batch embedding — process in chunks
    all_embeddings: list[list[float]] = []
    total_batches = (len(prepared) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[batch_idx : batch_idx + BATCH_SIZE]
        texts = [item['input_text'] for item in batch]

        current_batch_num = batch_idx // BATCH_SIZE + 1
        print(f'  Batch {current_batch_num}/{total_batches} ({len(texts)} predictions)...')

        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)

    print(f'\nGenerated {len(all_embeddings)} embeddings')
    print(f'Embedding dimension: {len(all_embeddings[0])}')

    # Combine metadata + embedding + input_text into final structure
    final_data = [
        {
            'id': prepared[i]['id'],
            'metadata': prepared[i]['metadata'],
            'embedding': all_embeddings[i],
            'input_text': prepared[i]['input_text'],
        }
        for i in range(len(prepared))
    ]

    # Save as pickle (fast to load, preserves numpy-friendly float lists)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(final_data, f)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f'\nWrote {OUTPUT_PATH}')
    print(f'File size: {file_size_mb:.2f} MB')

    # Rough cost estimate: ~250 tokens per prediction × $0.02 per 1M
    approx_tokens = len(final_data) * 250
    approx_cost = approx_tokens * 0.02 / 1_000_000
    print(f'\nApprox tokens: {approx_tokens:,}')
    print(f'Approx cost: ${approx_cost:.4f}')


if __name__ == '__main__':
    main()