"""
Task B Evaluation
==================
Runs recommendation agent on held-out users and measures:
  - NDCG@10, NDCG@5   (ranking quality — 30 pts)
  - Hit Rate@10, @5
  - MRR
  - Cold-start handling (25 pts)

Usage:
  python data/run_evaluation_b.py --source yelp --limit 10
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
import random

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.user_profile import UserProfile, RatingProfile, StyleProfile, PreferenceProfile, BehaviouralProfile
from tasks.task_b import recommend, CandidateItem, build_cold_start_profile
from evaluation.metrics import Evaluator

PREPARED_DIR = BASE_DIR / "datasets" / "prepared"
CACHE_DIR    = BASE_DIR / "datasets" / "cache"
RESULTS_DIR  = BASE_DIR / "datasets" / "results"

RANDOM_SEED  = 42


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


def build_candidate_pool(
    user_id: str,
    test_reviews: list,
    items_map: dict,
    pool_size: int = 50,
    seed: int = 42,
) -> tuple:
    """
    Build a candidate pool for a user containing:
    - All items the user actually reviewed in test set (ground truth positives)
    - Random negative items they haven't reviewed

    Returns: (candidates, relevant_item_ids)
    """
    random.seed(seed)

    # Ground truth: items the user rated highly in test (4+ stars = relevant)
    relevant_ids = {
        r["item_id"] for r in test_reviews
        if r.get("rating", 0) >= 4.0 and r["item_id"] in items_map
    }

    # All test item ids (positive)
    test_item_ids = {r["item_id"] for r in test_reviews if r["item_id"] in items_map}

    # Sample negatives from catalog (items user hasn't reviewed)
    all_item_ids = list(items_map.keys())
    negatives = [iid for iid in all_item_ids if iid not in test_item_ids]
    random.shuffle(negatives)
    negative_sample = negatives[:max(0, pool_size - len(test_item_ids))]

    # Build candidate list
    candidate_ids = list(test_item_ids) + negative_sample
    random.shuffle(candidate_ids)

    candidates = []
    for iid in candidate_ids:
        item = items_map.get(iid, {})
        if not item:
            continue
        candidates.append(CandidateItem(
            item_id=iid,
            item_name=item.get("item_name", ""),
            category=item.get("category", ""),
            description="",
            attributes=item.get("attributes", []),
            price_range=str(item.get("price_range", "")),
            avg_rating=float(item.get("stars", item.get("avg_rating", 0))),
            popularity=int(item.get("review_count", item.get("popularity", 0))),
        ))

    return candidates, list(relevant_ids)


def run_task_b_eval(source: str, limit: int = 10, delay: float = 8.0):
    print(f"\n{'='*60}", flush=True)
    print(f"TASK B EVALUATION — {source.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    test_file  = PREPARED_DIR / f"{source}_test.json"
    items_file = PREPARED_DIR / f"{source}_items.json"

    if not items_file.exists():
        print(f"  ERROR — no items catalog: {items_file}", flush=True)
        return

    with open(test_file, "r", encoding="utf-8") as f:
        all_test = json.load(f)

    with open(items_file, "r", encoding="utf-8") as f:
        items_list = json.load(f)
    items_map = {item["item_id"]: item for item in items_list}

    print(f"\n  Item catalog: {len(items_map):,} items", flush=True)

    # Group test reviews by user
    test_by_user = defaultdict(list)
    for r in all_test:
        test_by_user[r["user_id"]].append(r)

    # Find cached users with test reviews and at least 1 relevant item
    cached = [p.stem for p in (CACHE_DIR / source).glob("*.json")]
    eligible = []
    for uid in cached:
        if uid not in test_by_user:
            continue
        test_reviews = test_by_user[uid]
        has_relevant = any(
            r.get("rating", 0) >= 4.0 and r["item_id"] in items_map
            for r in test_reviews
        )
        if has_relevant:
            eligible.append(uid)
        if len(eligible) >= limit:
            break

    print(f"  Cached profiles: {len(cached)}", flush=True)
    print(f"  Eligible users:  {len(eligible)}", flush=True)
    print(f"  API calls:       {len(eligible)}", flush=True)
    print(f"  Estimated cost:  ~${len(eligible) * 0.004:.2f}", flush=True)

    # Cold-start test (3 users max, free from budget perspective)
    print(f"\n  [COLD-START TEST]", flush=True)
    cold_start_answers = {
        "q1": "I enjoy trying restaurants, cafes and local food spots",
        "q2": "3",
        "q3": "Poor service and dirty environment",
    }
    cold_profile = build_cold_start_profile("cold_user_test", cold_start_answers)
    print(f"  Cold-start profile built: {cold_profile.preferences.top_categories}", flush=True)

    print(f"\n  Running warm recommendations...\n", flush=True)

    ranked_lists = []
    relevant_sets = []

    for i, uid in enumerate(eligible, 1):
        profile = load_cached_profile(uid, source)
        if not profile:
            continue

        test_reviews = test_by_user[uid]
        candidates, relevant_ids = build_candidate_pool(
            uid, test_reviews, items_map,
            pool_size=50, seed=RANDOM_SEED + i,
        )

        if not candidates or not relevant_ids:
            print(f"  [{i}/{len(eligible)}] {uid[:12]}... SKIP (no candidates/relevant)", flush=True)
            continue

        print(f"  [{i}/{len(eligible)}] {uid[:12]}... "
              f"pool={len(candidates)} relevant={len(relevant_ids)}",
              end=" ", flush=True)

        try:
            result = recommend(
                profile=profile,
                candidates=candidates,
                top_k=min(10, len(candidates)),
            )

            ranked_item_ids = [r.item.item_id for r in result.recommendations]
            ranked_lists.append(ranked_item_ids)
            relevant_sets.append(relevant_ids)

            # Quick hit check
            hits = [iid for iid in ranked_item_ids[:10] if iid in relevant_ids]
            print(f"✓ hits@10={len(hits)}/{len(relevant_ids)}", flush=True)

        except Exception as e:
            print(f"✗ ERROR: {e}", flush=True)

        if i < len(eligible):
            time.sleep(delay)

    if not ranked_lists:
        print("\n  No results — check errors above.", flush=True)
        return

    # Evaluate
    print(f"\n  Evaluating {len(ranked_lists)} users...", flush=True)
    evaluator = Evaluator()
    results = evaluator.evaluate_task_b(
        ranked_lists, relevant_sets, k_values=[5, 10]
    )

    print(f"\n{'-'*60}", flush=True)
    print(results.summary([5, 10]), flush=True)
    print(f"{'-'*60}", flush=True)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source": source,
        "num_users": results.num_users,
        "ndcg_at_5":  results.ndcg_at_k.get(5, 0),
        "ndcg_at_10": results.ndcg_at_k.get(10, 0),
        "hit_rate_at_5":  results.hit_rate_at_k.get(5, 0),
        "hit_rate_at_10": results.hit_rate_at_k.get(10, 0),
        "mrr": results.mrr,
    }
    out_path = RESULTS_DIR / f"task_b_{source}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yelp", "amazon"], default="yelp")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--delay", type=float, default=8.0)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_task_b_eval(args.source, args.limit, args.delay)