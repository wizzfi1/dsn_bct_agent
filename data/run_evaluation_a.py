"""
Task A Evaluation

Runs review simulation on held-out test reviews and measures:
  - ROUGE-1, ROUGE-2, ROUGE-L  (text quality)
  - BERTScore F1               (semantic similarity)
  - RMSE                       (rating accuracy)

Uses cached profiles — no unnecessary API calls.
Each simulated review = 1 API call (~$0.002)

Usage:
  python data/run_evaluation_a.py --source yelp --limit 10
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.user_profile import UserProfile, RatingProfile, StyleProfile, PreferenceProfile, BehaviouralProfile
from tasks.task_a import simulate_review, ItemDetails
from evaluation.metrics import Evaluator

PREPARED_DIR = BASE_DIR / "datasets" / "prepared"
CACHE_DIR    = BASE_DIR / "datasets" / "cache"
RESULTS_DIR  = BASE_DIR / "datasets" / "results"


def load_cached_profile(user_id: str, source: str):
    path = CACHE_DIR / source / f"{user_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return UserProfile(
        user_id=data["user_id"],
        num_reviews=data["num_reviews"],
        rating=RatingProfile(**data["rating"]),
        style=StyleProfile(**data["style"]),
        preferences=PreferenceProfile(**data["preferences"]),
        behaviour=BehaviouralProfile(**data["behaviour"]),
        raw_summary=data["raw_summary"],
        confidence=data["confidence"],
    )


def run_task_a_eval(source: str, limit: int = 10, delay: float = 8.0):
    print(f"\n{'='*60}", flush=True)
    print(f"TASK A EVALUATION — {source.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load test and train reviews
    test_file  = PREPARED_DIR / f"{source}_test.json"
    train_file = PREPARED_DIR / f"{source}_train.json"
    items_file = PREPARED_DIR / f"{source}_items.json"

    with open(test_file, "r", encoding="utf-8") as f:
        all_test = json.load(f)

    with open(train_file, "r", encoding="utf-8") as f:
        all_train = json.load(f)

    # Load item catalog for enrichment
    items_map = {}
    if items_file.exists():
        with open(items_file, "r", encoding="utf-8") as f:
            items_list = json.load(f)
        items_map = {item["item_id"]: item for item in items_list}
        print(f"\n  Item catalog loaded: {len(items_map):,} items", flush=True)
    else:
        print(f"\n  No item catalog found — using basic item info only", flush=True)

    # Index train reviews by user (for few-shot examples)
    train_by_user = defaultdict(list)
    for r in all_train:
        train_by_user[r["user_id"]].append(r)

    # Group test reviews by user
    test_by_user = defaultdict(list)
    for r in all_test:
        test_by_user[r["user_id"]].append(r)

    # Find users that have cached profiles AND test reviews
    cached = [p.stem for p in (CACHE_DIR / source).glob("*.json")]
    eligible = [uid for uid in cached if uid in test_by_user][:limit]

    print(f"  Cached profiles:  {len(cached)}", flush=True)
    print(f"  Eligible users:   {len(eligible)}", flush=True)
    print(f"  API calls needed: {len(eligible)} (1 per user, 1 test review each)", flush=True)
    print(f"  Estimated cost:   ~${len(eligible) * 0.002:.2f}", flush=True)
    print(f"\n  Running simulations...\n", flush=True)

    predictions = []
    ground_truth = []

    for i, uid in enumerate(eligible, 1):
        profile = load_cached_profile(uid, source)
        if not profile:
            continue

        # Take first test review as ground truth
        test_review = test_by_user[uid][0]
        train_reviews = train_by_user.get(uid, [])

        # Enrich item with catalog data
        item_id = test_review.get("item_id", "")
        catalog_item = items_map.get(item_id, {})

        item = ItemDetails(
            item_id=item_id,
            item_name=test_review.get("item_name", "unknown"),
            category=test_review.get("category", ""),
            description="",
            attributes=catalog_item.get("attributes", []),
            price_range=str(catalog_item.get("price_range", "")),
            location=test_review.get("city", ""),
        )

        print(f"  [{i}/{len(eligible)}] {uid[:12]}... item: {item.item_name[:30]}",
              end=" ", flush=True)

        try:
            result = simulate_review(
                profile=profile,
                item=item,
                few_shot_reviews=train_reviews[-8:],
            )
            predictions.append({
                "text": result.review_text,
                "rating": result.predicted_rating,
            })
            ground_truth.append({
                "text": test_review["text"],
                "rating": test_review["rating"],
            })
            rating_diff = abs(result.predicted_rating - test_review["rating"])
            print(f"✓ predicted={result.predicted_rating} actual={test_review['rating']} "
                  f"diff={rating_diff:.1f}", flush=True)
        except Exception as e:
            print(f"✗ ERROR: {e}", flush=True)

        if i < len(eligible):
            time.sleep(delay)

    if not predictions:
        print("\n  No predictions made — check errors above.", flush=True)
        return

    # Evaluate
    print(f"\n  Evaluating {len(predictions)} predictions...", flush=True)
    evaluator = Evaluator()
    results = evaluator.evaluate_task_a(predictions, ground_truth)

    print(f"\n{'-'*60}", flush=True)
    print(results.summary(), flush=True)
    print(f"{'-'*60}", flush=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source": source,
        "num_samples": results.num_samples,
        "rouge1": results.rouge1,
        "rouge2": results.rouge2,
        "rougeL": results.rougeL,
        "bert_score_f1": results.bert_score_f1,
        "rmse": results.rmse,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }
    out_path = RESULTS_DIR / f"task_a_{source}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yelp", "amazon"], default="yelp")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of users to evaluate (default: 10)")
    parser.add_argument("--delay", type=float, default=8.0)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_task_a_eval(args.source, args.limit, args.delay)