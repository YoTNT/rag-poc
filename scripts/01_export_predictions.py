"""
Export verified predictions from DynamoDB to local JSON.

This is Step 1 of the RAG POC pipeline.

- Scans the production predictions table
- Filters to verified predictions only (has actual result to compare against)
- Keeps only fields needed for downstream embedding + analysis
- Writes to data/predictions.json

Usage:
    python scripts/01_export_predictions.py

Expected runtime: ~30 seconds
Output: data/predictions.json (~5-10 MB)
"""

import json
import os
from decimal import Decimal
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'guesshowmuch-production-predictions')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'predictions.json'

# Fields we keep from each DynamoDB item.
# Everything else (rawResponse, indicators details, etc.) is dropped
# to keep the JSON file small and focused on what RAG needs.
KEPT_FIELDS = [
    'id',
    'symbol',
    'predictedFor',
    'createdAt',
    'direction',
    'basePrice',
    'predictedPrice',
    'predictedChangePercent',
    'confidence',
    'confidenceLevel',
    'reasoning',
    'promptVersion',
    'verified',
    'actualPrice',
    'actualDirection',
    'wasCorrect',
    'indicatorsSnapshot',
]


def decimal_default(obj):
    """
    DynamoDB returns numbers as Decimal, which json.dump can't serialize.
    Convert to float (safe for prices, RSI, etc — no precision-critical
    values in our data).
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f'Cannot serialize {type(obj)}')


def scan_verified_predictions() -> list[dict]:
    """
    Scan the entire predictions table, keeping only verified ones.

    Uses pagination because DynamoDB scan returns at most 1MB per call.
    """
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    items: list[dict] = []
    last_key = None
    scan_count = 0

    while True:
        scan_count += 1
        kwargs = {
            'FilterExpression': 'verified = :v',
            'ExpressionAttributeValues': {':v': True},
        }
        if last_key:
            kwargs['ExclusiveStartKey'] = last_key

        response = table.scan(**kwargs)
        items.extend(response.get('Items', []))
        last_key = response.get('LastEvaluatedKey')

        print(f'  Scan {scan_count}: fetched {len(items)} verified predictions so far')

        if not last_key:
            break

    return items


def trim_fields(prediction: dict) -> dict:
    """Keep only the fields listed in KEPT_FIELDS."""
    return {k: prediction[k] for k in KEPT_FIELDS if k in prediction}


def main() -> None:
    print(f'Exporting verified predictions from {TABLE_NAME}...')

    raw_items = scan_verified_predictions()
    print(f'\nTotal verified predictions: {len(raw_items)}')

    trimmed = [trim_fields(item) for item in raw_items]

    # Ensure output directory exists (safety — should already exist)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(trimmed, f, default=decimal_default, indent=2)

    file_size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f'\nWrote {OUTPUT_PATH}')
    print(f'File size: {file_size_kb:.1f} KB')

    # Sanity checks — surface any data quality issues early
    with_reasoning = sum(1 for p in trimmed if p.get('reasoning'))
    with_indicators = sum(1 for p in trimmed if p.get('indicatorsSnapshot'))

    print(f'\nSanity checks:')
    print(f'  Predictions with reasoning:  {with_reasoning} / {len(trimmed)}')
    print(f'  Predictions with indicators: {with_indicators} / {len(trimmed)}')

    if with_reasoning < len(trimmed):
        missing = len(trimmed) - with_reasoning
        print(f'  ⚠️  {missing} predictions lack reasoning — will be skipped in embedding step')


if __name__ == '__main__':
    main()