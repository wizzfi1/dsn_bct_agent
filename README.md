# DSN × BCT LLM Agent Challenge 3.0
### A Unified LLM Agent Architecture for User Modeling and Personalised Recommendation

[![Task A](https://img.shields.io/badge/Task%20A-Review%20Simulation-2E5FA3)](https://dsn-bct-task-a-sjjw.onrender.com/docs)
[![Task B](https://img.shields.io/badge/Task%20B-Recommendation-1B7F3A)](https://dsn-bct-task-b-2gl8.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED)](https://docker.com)

---

## Overview

This system treats user modelling and personalised recommendation as two expressions of the same problem: **understanding the human behind the data**.

A single shared **User Profile Engine** extracts a rich, structured persona from a user's review history — capturing rating tendencies, writing style, cultural signals (including Nigerian Pidgin English), preferences, and behavioural patterns. Two task-specific agents are built on top:

- **Task A** — simulates reviews a user would write for unseen items, in their exact voice
- **Task B** — delivers personalised recommendations with explicit reasoning, handling both warm users and cold-start new users

---

## Live Demo

| Service | URL |
|---------|-----|
| Task A — Review Simulation | https://dsn-bct-task-a-sjjw.onrender.com |
| Task B — Recommendation | https://dsn-bct-task-b-2gl8.onrender.com |
| Interactive Demo Page | https://dsn-bct-task-a-sjjw.onrender.com |

---

## Results

| Metric | Score |
|--------|-------|
| RMSE (rating prediction) | **0.39** |
| BERTScore F1 | **0.85** |
| Hit Rate@10 | **1.00** |
| MRR | **0.67** |
| NDCG@10 | **0.50** |
| Cold-start NDCG@10 | **0.66** (vs 0.68 warm — gap of only 0.02) |

---

## Architecture

![System Architecture](https://raw.githubusercontent.com/wizzfi1/dsn_bct_agent/main/assets/arch_diagram.png)

*Figure 1: Shared User Profile Engine feeding both Task A and Task B agents, with cold-start elicitation path and live evaluation results.*

### How it works

```
Raw Review History (Yelp · Amazon · Goodreads)
              │
              ▼
┌─────────────────────────────────────────────┐
│            User Profile Engine              │  core/user_profile.py
│                                             │
│  RatingProfile    → mean · std · tendency   │
│  StyleProfile     → tone · Pidgin · length  │
│  PreferenceProfile→ loves · deal-breakers   │
│  BehaviouralProfile→ frequency · culture    │
│                                             │
│  🇳🇬  Nigerian Pidgin · Cultural Signals    │
│       Local References · Code-switching     │
│                                             │
│  ✓ Disk-cached after first extraction      │
│  ✓ ~1,500 tokens per user (one-time cost)  │
└─────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐     ┌──────────────────────┐
│     Task A      │     │       Task B          │
│ Review Simulation│     │   Recommendation     │
│                 │     │                      │
│ 8 few-shot      │     │ Reason before rank   │
│ exemplars       │     │ NDCG@10 scoring      │
│ Rating predict  │     │ Cold-start support   │
│ Voice mimicry   │     │ Multi-turn sessions  │
│                 │     │                      │
│ POST /simulate  │     │ POST /recommend/warm │
│ Port :8000      │     │ Port :8001           │
└─────────────────┘     └──────────────────────┘

Cold-Start path (new users — no history needed):
  3 Questions → Synthetic Profile → Task B
  Q1: Interests  Q2: Rating anchor  Q3: Deal-breaker
  Result: NDCG@10 within 0.02 of full warm profiles
```

For cold-start users (no review history), a **3-question elicitation protocol** bootstraps a complete profile from minimal input — achieving NDCG@10 within 0.02 of full warm profiles.

---

## Project Structure

```
dsn_bct_agent/
├── core/
│   └── user_profile.py        # Shared User Profile Engine
├── tasks/
│   ├── task_a.py              # Task A: Review Simulation Agent + FastAPI
│   └── task_b.py              # Task B: Recommendation Agent + FastAPI
├── data/
│   ├── loaders.py             # Yelp / Amazon / Goodreads dataset loaders
│   ├── prepare_data.py        # One-time dataset preparation (zero API cost)
│   ├── run_profiles.py        # Build and cache user profiles
│   ├── run_evaluation_a.py    # Task A evaluation (ROUGE, BERTScore, RMSE)
│   ├── run_evaluation_b.py    # Task B evaluation (NDCG, Hit Rate, MRR)
│   └── run_ablations.py       # All 3 ablation studies
├── evaluation/
│   └── metrics.py             # ROUGE, BERTScore, RMSE, NDCG, Hit Rate, MRR
├── tests/
│   └── test_end_to_end.py     # Full pipeline smoke test
├── Dockerfile.task_a          # Multi-stage Docker build for Task A
├── Dockerfile.task_b          # Multi-stage Docker build for Task B
├── docker-compose.yml         # Run both containers together
└── requirements.txt
```

---

## Quickstart — Docker (Recommended)

This is the fastest way to run both tasks. Judges should use this method.

**Prerequisites:** Docker installed, Anthropic API key

### 1. Clone the repository

```bash
git clone https://github.com/wizzfi1/dsn_bct_agent.git
cd dsn-bct-agent
```

### 2. Set your API key

```bash
# Linux / Mac
export ANTHROPIC_API_KEY=your-key-here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-key-here"
```

### 3. Build the images

```bash
docker build -f Dockerfile.task_a -t dsn-task-a:latest .
docker build -f Dockerfile.task_b -t dsn-task-b:latest .
```

### 4. Run Task A

```bash
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY dsn-task-a:latest
```

### 5. Run Task B (new terminal)

```bash
docker run -p 8001:8001 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY dsn-task-b:latest
```

### 6. Verify both are running

```bash
curl http://localhost:8000/health   # {"status":"ok","task":"A"}
curl http://localhost:8001/health   # {"status":"ok","task":"B"}
```

---

## API Reference

### Task A — Review Simulation

**Endpoint:** `POST http://localhost:8000/simulate`

Simulates a user review for an unseen item based on their review history.

**Request:**
```json
{
  "user_id": "user_001",
  "review_history": [
    {
      "text": "The jollof rice was perfect and service was fast!",
      "rating": 5.0,
      "category": "Nigerian Cuisine"
    },
    {
      "text": "Good food but the wait was too long abeg.",
      "rating": 3.0,
      "category": "Fast Food"
    }
  ],
  "item": {
    "item_id": "biz_001",
    "item_name": "Mama Cass Restaurant",
    "category": "Nigerian Cuisine",
    "attributes": ["local food", "affordable", "pepper soup"],
    "price_range": "affordable"
  },
  "include_reasoning": true
}
```

**Response:**
```json
{
  "user_id": "user_001",
  "item_id": "biz_001",
  "predicted_rating": 5.0,
  "review_text": "The pepper soup sweet die, will definitely come back!",
  "confidence": "high",
  "reasoning": "User consistently rates Nigerian cuisine highly and values affordability..."
}
```

---

### Task B — Recommendation (Warm User)

**Endpoint:** `POST http://localhost:8001/recommend/warm`

Delivers personalised recommendations for a user with review history.

**Request:**
```json
{
  "user_id": "user_001",
  "review_history": [
    {"text": "Amazing suya spot!", "rating": 5.0, "category": "Nigerian Cuisine"},
    {"text": "Overpriced and slow service.", "rating": 2.0, "category": "Fine Dining"}
  ],
  "candidates": [
    {"item_id": "c001", "item_name": "Bukka Hut", "category": "Nigerian Cuisine",
     "attributes": ["local food", "affordable"], "avg_rating": 4.7, "popularity": 850},
    {"item_id": "c002", "item_name": "Sky Lounge", "category": "Bar",
     "attributes": ["expensive", "slow service"], "avg_rating": 3.5, "popularity": 200}
  ],
  "top_k": 5,
  "context": "Looking for dinner tonight"
}
```

---

### Task B — Recommendation (Cold-Start)

**Endpoint:** `POST http://localhost:8001/recommend/cold-start`

Delivers personalised recommendations for a brand new user with no history.

**Request:**
```json
{
  "user_id": "new_user_001",
  "elicitation_answers": {
    "q1": "I love Nigerian food especially jollof rice and pepper soup",
    "q2": "3",
    "q3": "Bad service and overpriced food"
  },
  "candidates": [
    {"item_id": "c001", "item_name": "Bukka Hut", "category": "Nigerian Cuisine",
     "attributes": ["local food", "affordable"], "avg_rating": 4.7, "popularity": 850},
    {"item_id": "c002", "item_name": "Mama Put Central", "category": "Street Food",
     "attributes": ["affordable", "pepper soup"], "avg_rating": 4.5, "popularity": 2000}
  ],
  "top_k": 3
}
```

**Get elicitation questions:**
```bash
curl http://localhost:8001/cold-start/questions
```

---

### Interactive API Docs

Both containers expose auto-generated Swagger docs:
- Task A: http://localhost:8000/docs
- Task B: http://localhost:8001/docs

---

## Quickstart — Python (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-key-here   # Linux/Mac
$env:ANTHROPIC_API_KEY="your-key-here"  # Windows PowerShell

# Run end-to-end smoke test (uses built-in sample data)
python tests/test_end_to_end.py

# Run Task A API locally
uvicorn tasks.task_a:create_app --factory --host 0.0.0.0 --port 8000

# Run Task B API locally
uvicorn tasks.task_b:create_app --factory --host 0.0.0.0 --port 8001
```

---

## Reproducing Evaluation Results

### Step 1 — Prepare datasets (zero API cost)

Download datasets:
- **Yelp:** https://www.yelp.com/dataset
- **Amazon:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Goodreads:** https://mengtingwan.github.io/data/goodreads.html

Place in `datasets/` folder, then run:

```bash
python data/prepare_data.py
```

### Step 2 — Build user profiles

```bash
# Build 20 Yelp profiles (~$0.06)
python data/run_profiles.py --source yelp --limit 20 --delay 8

# Build 20 Amazon profiles (~$0.06)
python data/run_profiles.py --source amazon --limit 20 --delay 8
```

### Step 3 — Run evaluations

```bash
# Task A evaluation
python data/run_evaluation_a.py --source yelp --limit 10 --delay 8

# Task B evaluation
python data/run_evaluation_b.py --source yelp --limit 10 --delay 8
```

### Step 4 — Run ablation studies

```bash
python data/run_ablations.py --source yelp --limit 5 --delay 8 --ablation all
```

Results are saved to `datasets/results/`.

---

## Nigerian Localisation

The system explicitly detects and adapts to Nigerian cultural context:

| Signal | Examples | System Response |
|--------|----------|----------------|
| Nigerian Pidgin English | abeg, na, e don fall, sweet die, oya | Generates reviews with natural Pidgin |
| Local food references | jollof rice, suya, egusi, pounded yam | Raises cultural relevance in Task B |
| Consumer culture cues | value for money, customer is king | Adapts recommendation explanations |
| Social patterns | family dining, community references | Weights social-friendly venues higher |

**Example output** (detected Pidgin from 3 reviews, generated authentic response):
> *"The pepper soup sweet die, will definitely come back!"*

---

## Datasets Used

| Dataset | Source | Used For |
|---------|--------|----------|
| Yelp Academic Dataset | yelp.com/dataset | Task A + Task B (primary) |
| Amazon Reviews 2023 — Food & Grocery | HuggingFace | Task B cross-domain |
| Goodreads Reviews | mengtingwan.github.io | Task B cross-domain |

All datasets are used with appropriate disclosure per competition rules.

---

## Key Design Decisions

**Why one shared profile engine?** Both tasks require the same understanding of the user. Building two models would mean paying for the same extraction twice and risking inconsistency. The unified `UserProfile` ensures both agents reason from identical user understanding.

**Why disk caching?** Profile extraction costs ~1,500 input tokens per user. Caching ensures repeated evaluation and API serving never re-extract the same profile — keeping API costs minimal during development and evaluation.

**Why 3-question cold-start?** Ablation 3 proves the elicitation protocol achieves NDCG@10 within 0.02 of full warm profiles. The deal-breaker question alone provides strong negative preference signal that popularity-based baselines completely miss.

---

## Requirements

```
anthropic>=0.40.0
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.0.0
rouge-score>=0.1.2
bert-score>=0.3.13
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
sentence-transformers>=3.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
```

---

## License

Submitted for DSN × BCT LLM Agent Challenge 3.0. All datasets used with appropriate disclosure.