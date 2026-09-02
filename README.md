# rag-poc

RAG (Retrieval-Augmented Generation) proof-of-concept over the
[GuessHowMuch](https://github.com/YoTNT/GuessHowMuch) prediction
reasoning corpus.

**Goal:** validate whether retrieving similar past predictions at
inference time provides meaningful signal, before committing to a
production RAG microservice (Phase 1).

**Result (TL;DR):** Yes — indicator-only embeddings show a **+24.6 pp
accuracy delta** between high-context and low-context queries across
710 predictions. Recommendation: **proceed to Phase 1** with an
indicator-first embedding strategy.

---

## Motivation

GuessHowMuch has accumulated a meaningful body of prediction data —
710 verified predictions across Mega 7 + reward-tier stocks as of
this writing. Each prediction includes the Claude-generated reasoning
that led to the call, along with the actual outcome once verified the
next trading day.

The current system already learns from historical mistakes: a
post-mortem worker analyzes wrong predictions and feeds abstract
lessons into the prompt evolution pipeline (v1 → v4-bollinger-aware).
This is prompt-level, batch-mode learning — accumulated wisdom gets
compiled into a new prompt version every few weeks.

But market situations repeat. Similar technical setups + similar
news sentiment + similar price structures show up again and again,
and Claude's judgment on those setups also follows patterns. With
enough historical data, those patterns become retrievable.

This POC explores whether retrieving concrete past cases at
prediction time — not just relying on abstract compiled lessons —
provides additional signal. It's meant as a **complement** to the
existing prompt-evolution loop, not a replacement:

| Mechanism | Granularity | Timing | Output |
|---|---|---|---|
| Post-mortem + Prompt evolution (existing) | Prompt-version level | Batch (weeks) | Abstract rules baked into system prompt |
| **RAG (proposed)** | Per-prediction level | Real-time (per call) | Concrete similar past cases injected into context |

The goal of this POC is to validate the approach and produce
data-driven direction for a production-grade RAG microservice.

---

## Repo overview

Five sequential scripts form the POC pipeline:

```
scripts/
├── 01_export_predictions.py       Pull verified predictions from DynamoDB → JSON
├── 02_generate_embeddings.py      Embed each prediction via OpenAI → pickle
├── 03_search_similar.py           Interactive similarity-search CLI (eyeball retrieval quality)
├── 04_analyze_patterns.py         Systematic signal check across the full corpus
└── 05_compare_strategies.py       Compare 3 embedding strategies to find the best
```

**Data flow:**

```
DynamoDB predictions table
    ↓ (script 01)
data/predictions.json  (710 verified predictions)
    ↓ (script 02)
data/embeddings.pkl  (710 × 1536-dim OpenAI embeddings + metadata)
    ↓ (script 03)                    (script 04)              (script 05)
Manual eyeball QA          Systematic pattern analysis    Strategy comparison
                                    ↓                              ↓
                          data/pattern_analysis.csv   data/strategy_comparison.csv
```

**Design choices:**

- **Embedding provider:** OpenAI `text-embedding-3-small` (1536 dim,
  $0.02 per 1M tokens). Total POC cost: ~$0.01.
- **Storage:** local pickle files. No vector DB — 710 embeddings fit
  in ~10MB, in-memory cosine similarity is <1 sec. Vector DB is a
  Phase 1 concern.
- **Similarity metric:** cosine similarity (industry standard for
  normalized embeddings).
- **Analysis window:** top-K=20 similar predictions per query as
  "context accuracy" signal.

---

## Key findings

### Finding 1: RAG has meaningful predictive signal

Across all 710 predictions, bucket queries by their similar-context
accuracy (top-20 neighbors):

| Bucket | Query count | Query accuracy | vs Baseline (42.5%) |
|---|---|---|---|
| High-context (similar-20 acc > 60%) | 50 | 48.0% | +5.5 pp |
| Mid-context | 450 | 47.8% | +5.2 pp |
| **Low-context** (similar-20 acc < 40%) | **210** | **30.0%** | **−12.5 pp** |

**Key delta: +18 pp between high and low context buckets.**

The strongest signal is on the downside: when RAG finds that past
similar predictions were largely wrong (< 40% accurate), the current
prediction is also disproportionately likely to be wrong (30% vs
42.5% baseline). This is the exact scenario where RAG can warn the
model to lower confidence or reconsider direction.

### Finding 2: Indicator-only embeddings outperform structured embeddings

Compared three embedding-input strategies on the same corpus:

| Strategy | Baseline | High-context accuracy | Low-context accuracy | Delta (pp) |
|---|---|---|---|---|
| `reasoning_only` | 42.5% | 55.4% (n=56) | 40.1% (n=269) | +15.2 |
| **`indicators_only`** | 42.5% | **55.6%** (n=54) | **31.0%** (n=242) | **+24.6** |
| `structured` (reasoning + indicators + metadata) | 42.5% | 50.0% (n=50) | 30.7% (n=212) | +19.3 |

**Winner: `indicators_only` by a wide margin.**

This was **counter to the initial hypothesis** that combining
reasoning + indicators + metadata ("more context = better signal")
would win. Likely explanation: reasoning text carries Claude's
stylistic signature (prompt-version-specific phrasing like "Layer 1
(Catalyst Check)"), which causes embeddings to cluster by writing
style rather than by market situation. Pure numeric indicators
capture the objective market state more directly.

**Implication for Phase 1:** use indicator-first embedding as the
primary retrieval strategy, with reasoning-based embedding as an
optional secondary vector.

### Finding 3: Per-prompt-version accuracy (bonus observation)

| Prompt version | Predictions | Baseline accuracy |
|---|---|---|
| v2-technical-focus | 58 | 53.4% |
| v1-postmortem | 233 | 48.5% |
| v3-context-aware | 135 | 37.8% |
| v4-bollinger-aware | 279 | 37.6% |

Newer prompt versions (v3, v4) are performing worse than earlier
ones (v1, v2), despite being "more sophisticated." This is
tangential to RAG validation but worth flagging for GuessHowMuch's
prompt evolution roadmap — the added Layer 1/2/3 protocol in v4 may
be over-constraining Claude's reasoning.

(Small caveat: v2 sample size is only 58, so its 53.4% has wider
confidence bounds than the others.)

---

## Reproducing the results

Prerequisites:
- Python 3.10+
- AWS credentials with read access to the GuessHowMuch predictions
  table
- OpenAI API key with a small credit balance (< $1 needed)

Setup:

```bash
git clone https://github.com/YoTNT/rag-poc.git
cd rag-poc

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your OpenAI key and AWS region
```

Run the pipeline:

```bash
# Step 1: pull verified predictions from DynamoDB
python scripts/01_export_predictions.py

# Step 2: generate embeddings (main strategy)
python scripts/02_generate_embeddings.py

# Step 3: interactively explore retrieval quality
python scripts/03_search_similar.py --random --n 5

# Step 4: systematic signal analysis
python scripts/04_analyze_patterns.py

# Step 5: compare embedding strategies
python scripts/05_compare_strategies.py
```

Total runtime: ~10 minutes end-to-end. Total OpenAI cost: ~$0.01.

---

## Next steps (Phase 1 plan)

**Decision:** proceed to Phase 1 — a production RAG microservice
integrated into the GuessHowMuch prediction pipeline.

**Architecture:** independent Python microservice, separate from the
existing `sentiment-engine` service. Rationale: RAG (semantic text
retrieval) and sentiment engine (objective market signal
classification) operate on different data types, have different
failure modes, and evolve on different cadences — separate services
preserve bounded context and allow independent scaling.

**Proposed stack:**

- **Language / framework:** Python 3.12 + FastAPI (matches sentiment-
  engine pattern; reuses Terraform module and CI patterns)
- **Vector store:** to be evaluated in Phase 1 — candidates are
  pgvector on RDS, Pinecone, or DynamoDB S3 Vectors. Local pickle is
  POC-only and doesn't survive Phase 1.
- **Embedding strategy:** `indicators_only` as primary retrieval
  vector (based on POC finding 2). Optional secondary
  reasoning-based vector for future hybrid retrieval.
- **Deployment:** ECS Fargate on private VPC (reuses
  sentiment-engine's IAM + Terraform module)
- **New endpoints:**
  - `POST /rag/similar` — return top-K similar prediction IDs for a
    given query prediction
  - `POST /rag/embed` — embed a new prediction and add it to the
    index (called after every new prediction is written to the main
    predictions table)

**Integration with core service:**

- Add an optional RAG step to `predictionService` — feature-flag
  controlled so it can be turned off instantly
- Injected into the Claude prompt as a "Historical Context" section:
  "Here are 5 past predictions with similar market conditions:
  X wrong, Y correct, average confidence Z"
- A/B test: 50% of new predictions use RAG-injected context, 50%
  don't; compare accuracy after 4 weeks

**Success criteria for Phase 1:**

- End-to-end latency (embed query + search + inject) < 500ms
- Monthly infrastructure cost < $10 (embedding API + vector DB)
- A/B test shows accuracy delta ≥ 3 pp in favor of RAG group after
  4 weeks (statistically weaker than POC's 24.6pp, but the POC
  compared bucket extremes while A/B measures average effect)

**Estimated timeline:** 2–3 weeks (Week 1: infrastructure + repo,
Week 2: service API + vector DB migration, Week 3: core-service
integration + A/B setup).

**Future directions beyond Phase 1:**

- Hybrid retrieval combining indicator + reasoning embeddings
- User-facing "similar past predictions" feature on the stock detail
  page (product surface for the same infrastructure)
- Cross-symbol clustering (POC showed cross-symbol similarity can be
  meaningful, e.g., TSLA DOWN patterns predicted AAPL DOWN outcomes)

---

## Related repos

- [GuessHowMuch](https://github.com/YoTNT/GuessHowMuch) — the main
  Node.js prediction service that this POC's data comes from
- [GuessHowMuch-web](https://github.com/YoTNT/GuessHowMuch-web) —
  React/TypeScript frontend
- [GuessHowMuch-infra](https://github.com/YoTNT/GuessHowMuch-infra) —
  Terraform infrastructure

---

## Status

POC complete. Repository preserved as reference for the Phase 1
production service (to be built in a separate repo,
`rag-engine` or similar).

Last updated: 2026-09-01
