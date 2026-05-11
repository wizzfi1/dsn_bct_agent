# DSN X BCT LLM Agent Challenge — Submission

## Architecture Overview

```
dsn_bct_agent/
├── core/
│   └── user_profile.py      # Shared User Profile Engine (foundation of both tasks)
├── data/
│   └── loaders.py           # Yelp / Amazon / Goodreads dataset loaders + sample data
├── tasks/
│   ├── task_a.py            # Task A: Review Simulation Agent + FastAPI endpoint
│   └── task_b.py            # Task B: Recommendation Agent + FastAPI endpoint
├── evaluation/
│   └── metrics.py           # ROUGE, BERTScore, RMSE, NDCG, Hit Rate, MRR
├── tests/
│   └── test_end_to_end.py   # Full pipeline smoke test
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=your_key_here

# 3. Run the end-to-end test (uses built-in sample data)
python tests/test_end_to_end.py
```

## Running the APIs

### Task A — Review Simulation
```bash
uvicorn tasks.task_a:create_app --factory --host 0.0.0.0 --port 8000
```

**POST /simulate**
```json
{
  "user_id": "user_ng_001",
  "review_history": [
    {"text": "The jollof rice was perfect!", "rating": 5.0, "category": "Nigerian Cuisine"}
  ],
  "item": {
    "item_id": "biz_001",
    "item_name": "Bukka Hut VI",
    "category": "Nigerian Cuisine",
    "attributes": ["affordable", "local food", "outdoor seating"]
  }
}
```

### Task B — Recommendation
```bash
uvicorn tasks.task_b:create_app --factory --host 0.0.0.0 --port 8001
```

**POST /recommend/warm** — for users with review history

**POST /recommend/cold-start** — for new users
```json
{
  "user_id": "new_user",
  "elicitation_answers": {
    "q1": "I love Nigerian food and fast casual restaurants",
    "q2": "3",
    "q3": "Bad service and overpriced food"
  },
  "candidates": [...]
}
```

**GET /cold-start/questions** — get the 3 elicitation questions

## Docker

```bash
# Build
docker build -t dsn-bct-agent .

# Run Task A
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 dsn-bct-agent

# Run Task B
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -e TASK=B -p 8001:8001 dsn-bct-agent
```

## Evaluation

```python
from evaluation.metrics import Evaluator

evaluator = Evaluator()

# Task A
results = evaluator.evaluate_task_a(predictions, ground_truth)
print(results.summary())

# Task B
results = evaluator.evaluate_task_b(ranked_lists, relevant_items, k_values=[5, 10])
print(results.summary([5, 10]))
```

## Dataset

Download from:
- **Yelp**: https://www.yelp.com/dataset
- **Amazon Reviews**: https://nijianmo.github.io/amazon/index.html
- **Goodreads**: https://mengtingwan.github.io/data/goodreads.html

Update paths in `data/loaders.py` accordingly.

## Nigerian Localisation

The User Profile Engine detects Nigerian Pidgin English and cultural signals
in review histories. The simulation agent amplifies these when generating
reviews for users with Nigerian cultural context, addressing the explicit
bonus criterion in the competition brief.
