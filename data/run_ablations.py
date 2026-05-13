"""
Ablation Studies

Runs three targeted ablations to prove each component contributes.

Ablation 1: Task A without few-shot examples
Ablation 2: Task B without UserProfile (popularity baseline)
Ablation 3: Task B cold-start vs warm profile

Usage:
  python data/run_ablations.py --source yelp --limit 5
"""

import json
import os
import sys
import argparse
import time
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.user_profile import UserProfile, RatingProfile, StyleProfile, PreferenceProfile, BehaviouralProfile
from tasks.task_a import simulate_review, ItemDetails
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


def build_candidate_pool(user_id, test_reviews, items_map, pool_size=20, seed=42):
    random.seed(seed)
    relevant_ids = {
        r["item_id"] for r in test_reviews
        if r.get("rating", 0) >= 4.0 and r["item_id"] in items_map
    }
    test_item_ids = {r["item_id"] for r in test_reviews if r["item_id"] in items_map}
    negatives = [iid for iid in items_map.keys() if iid not in test_item_ids]
    random.shuffle(negatives)
    negative_sample = negatives[:max(0, pool_size - len(test_item_ids))]
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


def popularity_rank(candidates, top_k):
    """Rank purely by avg_rating * log(popularity+1) — no LLM, no user profile."""
    import math
    scored = [
        (c, c.avg_rating * math.log(c.popularity + 1))
        for c in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c.item_id for c, _ in scored[:top_k]]


# Ablation 1: Task A without few-shot examples

def ablation_1_no_few_shot(source: str, limit: int, delay: float):
    print(f"\n{'='*60}", flush=True)
    print("ABLATION 1: Task A WITHOUT few-shot examples", flush=True)
    print(f"{'='*60}", flush=True)
    print("Hypothesis: removing real review examples hurts style mimicry", flush=True)

    test_file  = PREPARED_DIR / f"{source}_test.json"
    train_file = PREPARED_DIR / f"{source}_train.json"
    items_file = PREPARED_DIR / f"{source}_items.json"

    with open(test_file, "r", encoding="utf-8") as f:
        all_test = json.load(f)
    with open(train_file, "r", encoding="utf-8") as f:
        all_train = json.load(f)

    items_map = {}
    if items_file.exists():
        with open(items_file, "r", encoding="utf-8") as f:
            items_map = {i["item_id"]: i for i in json.load(f)}

    train_by_user = defaultdict(list)
    for r in all_train:
        train_by_user[r["user_id"]].append(r)

    test_by_user = defaultdict(list)
    for r in all_test:
        test_by_user[r["user_id"]].append(r)

    cached = [p.stem for p in (CACHE_DIR / source).glob("*.json")]
    eligible = [uid for uid in cached if uid in test_by_user][:limit]

    print(f"\n  Users: {len(eligible)} | API calls: {len(eligible)}", flush=True)
    print(f"  Estimated cost: ~${len(eligible) * 0.002:.2f}\n", flush=True)

    predictions_with    = []
    predictions_without = []
    ground_truth        = []

    evaluator = Evaluator()

    for i, uid in enumerate(eligible, 1):
        profile = load_cached_profile(uid, source)
        if not profile:
            continue

        test_review  = test_by_user[uid][0]
        train_reviews = train_by_user.get(uid, [])
        item_id = test_review.get("item_id", "")
        catalog = items_map.get(item_id, {})

        item = ItemDetails(
            item_id=item_id,
            item_name=test_review.get("item_name", "unknown"),
            category=test_review.get("category", ""),
            attributes=catalog.get("attributes", []),
            price_range=str(catalog.get("price_range", "")),
            location=test_review.get("city", ""),
        )

        print(f"  [{i}/{len(eligible)}] {uid[:12]}...", end=" ", flush=True)

        try:
            # WITH few-shot (normal)
            result_with = simulate_review(
                profile=profile,
                item=item,
                few_shot_reviews=train_reviews[-8:],
            )
            time.sleep(delay)

            # WITHOUT few-shot (ablation)
            result_without = simulate_review(
                profile=profile,
                item=item,
                few_shot_reviews=None,  # <-- key difference
            )

            predictions_with.append({"text": result_with.review_text, "rating": result_with.predicted_rating})
            predictions_without.append({"text": result_without.review_text, "rating": result_without.predicted_rating})
            ground_truth.append({"text": test_review["text"], "rating": test_review["rating"]})

            print(f"✓ rating_with={result_with.predicted_rating} "
                  f"rating_without={result_without.predicted_rating} "
                  f"actual={test_review['rating']}", flush=True)

        except Exception as e:
            print(f"✗ ERROR: {e}", flush=True)

        if i < len(eligible):
            time.sleep(delay)

    if not predictions_with:
        print("  No results.", flush=True)
        return {}

    r_with    = evaluator.evaluate_task_a(predictions_with, ground_truth)
    r_without = evaluator.evaluate_task_a(predictions_without, ground_truth)

    print(f"\n  {'Metric':<20} {'WITH few-shot':>15} {'WITHOUT few-shot':>18} {'Delta':>10}", flush=True)
    print(f"  {'-'*65}", flush=True)
    print(f"  {'ROUGE-1':<20} {r_with.rouge1:>15.4f} {r_without.rouge1:>18.4f} {r_without.rouge1 - r_with.rouge1:>+10.4f}", flush=True)
    print(f"  {'ROUGE-2':<20} {r_with.rouge2:>15.4f} {r_without.rouge2:>18.4f} {r_without.rouge2 - r_with.rouge2:>+10.4f}", flush=True)
    print(f"  {'ROUGE-L':<20} {r_with.rougeL:>15.4f} {r_without.rougeL:>18.4f} {r_without.rougeL - r_with.rougeL:>+10.4f}", flush=True)
    print(f"  {'BERTScore F1':<20} {r_with.bert_score_f1:>15.4f} {r_without.bert_score_f1:>18.4f} {r_without.bert_score_f1 - r_with.bert_score_f1:>+10.4f}", flush=True)
    print(f"  {'RMSE':<20} {r_with.rmse:>15.4f} {r_without.rmse:>18.4f} {r_without.rmse - r_with.rmse:>+10.4f}", flush=True)

    return {
        "with_few_shot":    {"rouge1": r_with.rouge1, "rouge2": r_with.rouge2, "rougeL": r_with.rougeL, "bert_score": r_with.bert_score_f1, "rmse": r_with.rmse},
        "without_few_shot": {"rouge1": r_without.rouge1, "rouge2": r_without.rouge2, "rougeL": r_without.rougeL, "bert_score": r_without.bert_score_f1, "rmse": r_without.rmse},
    }


# Ablation 2: Task B without UserProfile (popularity baseline)

def ablation_2_no_profile(source: str, limit: int):
    print(f"\n{'='*60}", flush=True)
    print("ABLATION 2: Task B WITHOUT UserProfile (popularity baseline)", flush=True)
    print(f"{'='*60}", flush=True)
    print("Hypothesis: user-aware ranking outperforms popularity ranking", flush=True)
    print("Cost: $0.00 — no API calls needed\n", flush=True)

    test_file  = PREPARED_DIR / f"{source}_test.json"
    items_file = PREPARED_DIR / f"{source}_items.json"

    with open(test_file, "r", encoding="utf-8") as f:
        all_test = json.load(f)
    with open(items_file, "r", encoding="utf-8") as f:
        items_map = {i["item_id"]: i for i in json.load(f)}

    test_by_user = defaultdict(list)
    for r in all_test:
        test_by_user[r["user_id"]].append(r)

    cached = [p.stem for p in (CACHE_DIR / source).glob("*.json")]

    # Load saved Task B results (agent ranking)
    results_file = RESULTS_DIR / f"task_b_{source}.json"
    if not results_file.exists():
        print(f"  ERROR — run run_evaluation_b.py first to get agent results", flush=True)
        return {}

    with open(results_file, "r", encoding="utf-8") as f:
        agent_results = json.load(f)

    # Now compute popularity baseline for same users
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

    pop_ranked_lists = []
    pop_relevant_sets = []

    for i, uid in enumerate(eligible, 1):
        test_reviews = test_by_user[uid]
        candidates, relevant_ids = build_candidate_pool(
            uid, test_reviews, items_map,
            pool_size=20, seed=RANDOM_SEED + i,
        )
        if not candidates or not relevant_ids:
            continue

        pop_ranking = popularity_rank(candidates, top_k=10)
        pop_ranked_lists.append(pop_ranking)
        pop_relevant_sets.append(relevant_ids)

        hits = [iid for iid in pop_ranking[:10] if iid in relevant_ids]
        print(f"  [{i}/{len(eligible)}] {uid[:12]}... popularity hits@10={len(hits)}/{len(relevant_ids)}", flush=True)

    evaluator = Evaluator()
    pop_eval = evaluator.evaluate_task_b(pop_ranked_lists, pop_relevant_sets, k_values=[5, 10])

    print(f"\n  {'Metric':<20} {'Agent (LLM)':>15} {'Popularity':>15} {'Delta':>10}", flush=True)
    print(f"  {'-'*62}", flush=True)
    print(f"  {'NDCG@10':<20} {agent_results['ndcg_at_10']:>15.4f} {pop_eval.ndcg_at_k[10]:>15.4f} {agent_results['ndcg_at_10'] - pop_eval.ndcg_at_k[10]:>+10.4f}", flush=True)
    print(f"  {'Hit Rate@10':<20} {agent_results['hit_rate_at_10']:>15.4f} {pop_eval.hit_rate_at_k[10]:>15.4f} {agent_results['hit_rate_at_10'] - pop_eval.hit_rate_at_k[10]:>+10.4f}", flush=True)
    print(f"  {'MRR':<20} {agent_results['mrr']:>15.4f} {pop_eval.mrr:>15.4f} {agent_results['mrr'] - pop_eval.mrr:>+10.4f}", flush=True)
    print(f"  {'NDCG@5':<20} {agent_results['ndcg_at_5']:>15.4f} {pop_eval.ndcg_at_k[5]:>15.4f} {agent_results['ndcg_at_5'] - pop_eval.ndcg_at_k[5]:>+10.4f}", flush=True)

    return {
        "agent":      {"ndcg_10": agent_results["ndcg_at_10"], "hit_10": agent_results["hit_rate_at_10"], "mrr": agent_results["mrr"]},
        "popularity": {"ndcg_10": pop_eval.ndcg_at_k[10], "hit_10": pop_eval.hit_rate_at_k[10], "mrr": pop_eval.mrr},
    }


# Ablation 3: Cold-start vs warm profile

def ablation_3_cold_vs_warm(source: str, limit: int, delay: float):
    print(f"\n{'='*60}", flush=True)
    print("ABLATION 3: Cold-start vs Warm profile", flush=True)
    print(f"{'='*60}", flush=True)
    print("Hypothesis: warm profiles outperform cold-start elicitation", flush=True)

    test_file  = PREPARED_DIR / f"{source}_test.json"
    items_file = PREPARED_DIR / f"{source}_items.json"

    with open(test_file, "r", encoding="utf-8") as f:
        all_test = json.load(f)
    with open(items_file, "r", encoding="utf-8") as f:
        items_map = {i["item_id"]: i for i in json.load(f)}

    test_by_user = defaultdict(list)
    for r in all_test:
        test_by_user[r["user_id"]].append(r)

    cached = [p.stem for p in (CACHE_DIR / source).glob("*.json")]
    eligible = []
    for uid in cached:
        if uid not in test_by_user:
            continue
        has_relevant = any(
            r.get("rating", 0) >= 4.0 and r["item_id"] in items_map
            for r in test_by_user[uid]
        )
        if has_relevant:
            eligible.append(uid)
        if len(eligible) >= limit:
            break

    print(f"\n  Users: {len(eligible)} | API calls: {len(eligible)} (cold-start profiles)", flush=True)
    print(f"  Estimated cost: ~${len(eligible) * 0.003:.2f}\n", flush=True)

    warm_ranked   = []
    cold_ranked   = []
    relevant_sets = []

    evaluator = Evaluator()

    # Generic cold-start answers (same for all users — simulates a brand new user)
    cold_answers = {
        "q1": "I enjoy restaurants, cafes, and trying new local food spots",
        "q2": "3",
        "q3": "Poor hygiene and rude staff",
    }

    for i, uid in enumerate(eligible, 1):
        warm_profile = load_cached_profile(uid, source)
        if not warm_profile:
            continue

        test_reviews = test_by_user[uid]
        candidates, relevant_ids = build_candidate_pool(
            uid, test_reviews, items_map,
            pool_size=20, seed=RANDOM_SEED + i,
        )
        if not candidates or not relevant_ids:
            continue

        print(f"  [{i}/{len(eligible)}] {uid[:12]}...", end=" ", flush=True)

        try:
            # Warm recommendation
            warm_result = recommend(
                profile=warm_profile,
                candidates=candidates,
                top_k=min(10, len(candidates)),
            )
            time.sleep(delay)

            # Cold-start recommendation
            cold_profile = build_cold_start_profile(
                user_id=f"cold_{uid}",
                elicitation_answers=cold_answers,
            )
            cold_result = recommend(
                profile=cold_profile,
                candidates=candidates,
                top_k=min(10, len(candidates)),
            )

            warm_ids = [r.item.item_id for r in warm_result.recommendations]
            cold_ids = [r.item.item_id for r in cold_result.recommendations]

            warm_ranked.append(warm_ids)
            cold_ranked.append(cold_ids)
            relevant_sets.append(relevant_ids)

            warm_hits = len([iid for iid in warm_ids[:10] if iid in relevant_ids])
            cold_hits = len([iid for iid in cold_ids[:10] if iid in relevant_ids])
            print(f"✓ warm_hits={warm_hits} cold_hits={cold_hits}", flush=True)

        except Exception as e:
            print(f"✗ ERROR: {e}", flush=True)

        if i < len(eligible):
            time.sleep(delay)

    if not warm_ranked:
        print("  No results.", flush=True)
        return {}

    warm_eval = evaluator.evaluate_task_b(warm_ranked, relevant_sets, k_values=[5, 10])
    cold_eval = evaluator.evaluate_task_b(cold_ranked, relevant_sets, k_values=[5, 10])

    print(f"\n  {'Metric':<20} {'Warm profile':>15} {'Cold-start':>15} {'Delta':>10}", flush=True)
    print(f"  {'-'*62}", flush=True)
    print(f"  {'NDCG@10':<20} {warm_eval.ndcg_at_k[10]:>15.4f} {cold_eval.ndcg_at_k[10]:>15.4f} {warm_eval.ndcg_at_k[10] - cold_eval.ndcg_at_k[10]:>+10.4f}", flush=True)
    print(f"  {'Hit Rate@10':<20} {warm_eval.hit_rate_at_k[10]:>15.4f} {cold_eval.hit_rate_at_k[10]:>15.4f} {warm_eval.hit_rate_at_k[10] - cold_eval.hit_rate_at_k[10]:>+10.4f}", flush=True)
    print(f"  {'MRR':<20} {warm_eval.mrr:>15.4f} {cold_eval.mrr:>15.4f} {warm_eval.mrr - cold_eval.mrr:>+10.4f}", flush=True)

    return {
        "warm": {"ndcg_10": warm_eval.ndcg_at_k[10], "hit_10": warm_eval.hit_rate_at_k[10], "mrr": warm_eval.mrr},
        "cold": {"ndcg_10": cold_eval.ndcg_at_k[10], "hit_10": cold_eval.hit_rate_at_k[10], "mrr": cold_eval.mrr},
    }


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yelp", "amazon"], default="yelp")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--delay", type=float, default=8.0)
    parser.add_argument("--ablation", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    all_results = {}

    if args.ablation in ("1", "all"):
        all_results["ablation_1"] = ablation_1_no_few_shot(
            args.source, args.limit, args.delay
        )

    if args.ablation in ("2", "all"):
        all_results["ablation_2"] = ablation_2_no_profile(
            args.source, args.limit
        )

    if args.ablation in ("3", "all"):
        all_results["ablation_3"] = ablation_3_cold_vs_warm(
            args.source, args.limit, args.delay
        )

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ablations_{args.source}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"All ablations complete. Results saved to: {out_path}", flush=True)
