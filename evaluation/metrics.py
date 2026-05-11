"""
Evaluation Harness
===================
Implements all metrics used in the hackathon scoring rubric.

Task A metrics:
  - ROUGE-1, ROUGE-2, ROUGE-L   (review text quality)
  - BERTScore F1                 (semantic similarity)
  - RMSE                         (rating accuracy)

Task B metrics:
  - NDCG@K                       (ranking quality)
  - Hit Rate@K                   (did the right item appear in top-K?)
  - MRR                          (mean reciprocal rank)

Usage:
  evaluator = Evaluator()
  results = evaluator.evaluate_task_a(predictions, ground_truth)
  results = evaluator.evaluate_task_b(ranked_lists, ground_truth)
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskAResults:
    rouge1: float
    rouge2: float
    rougeL: float
    bert_score_f1: Optional[float]    # None if bert_score not installed
    rmse: float
    num_samples: int

    def summary(self) -> str:
        lines = [
            f"Task A Evaluation ({self.num_samples} samples)",
            f"  ROUGE-1:      {self.rouge1:.4f}",
            f"  ROUGE-2:      {self.rouge2:.4f}",
            f"  ROUGE-L:      {self.rougeL:.4f}",
            f"  BERTScore F1: {self.bert_score_f1:.4f}" if self.bert_score_f1 else "  BERTScore:   not installed",
            f"  RMSE:         {self.rmse:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class TaskBResults:
    ndcg_at_k: dict[int, float]      # {k: ndcg_score}
    hit_rate_at_k: dict[int, float]  # {k: hit_rate}
    mrr: float
    num_users: int

    def summary(self, k_values: list[int] = None) -> str:
        if k_values is None:
            k_values = sorted(self.ndcg_at_k.keys())
        lines = [f"Task B Evaluation ({self.num_users} users)"]
        for k in k_values:
            lines.append(f"  NDCG@{k}:     {self.ndcg_at_k.get(k, 0):.4f}")
            lines.append(f"  Hit Rate@{k}: {self.hit_rate_at_k.get(k, 0):.4f}")
        lines.append(f"  MRR:          {self.mrr:.4f}")
        return "\n".join(lines)


class Evaluator:

    # -----------------------------------------------------------------------
    # Task A
    # -----------------------------------------------------------------------

    def evaluate_task_a(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
    ) -> TaskAResults:
        """
        predictions: list of {"text": str, "rating": float}
        ground_truth: list of {"text": str, "rating": float}
        """
        assert len(predictions) == len(ground_truth), "Lengths must match"

        pred_texts = [p["text"] for p in predictions]
        ref_texts  = [g["text"] for g in ground_truth]

        # ROUGE
        rouge1, rouge2, rougeL = self._compute_rouge(pred_texts, ref_texts)

        # BERTScore (optional)
        bert_f1 = self._compute_bert_score(pred_texts, ref_texts)

        # RMSE
        pred_ratings = [float(p.get("rating", 3.0)) for p in predictions]
        true_ratings = [float(g.get("rating", 3.0)) for g in ground_truth]
        rmse = self._compute_rmse(pred_ratings, true_ratings)

        return TaskAResults(
            rouge1=rouge1,
            rouge2=rouge2,
            rougeL=rougeL,
            bert_score_f1=bert_f1,
            rmse=rmse,
            num_samples=len(predictions),
        )

    def _compute_rouge(
        self,
        predictions: list[str],
        references: list[str],
    ) -> tuple[float, float, float]:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
            r1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores)
            r2 = sum(s["rouge2"].fmeasure for s in scores) / len(scores)
            rL = sum(s["rougeL"].fmeasure for s in scores) / len(scores)
            return r1, r2, rL
        except ImportError:
            print("Warning: rouge_score not installed. pip install rouge-score")
            return self._rouge_fallback(predictions, references)

    def _rouge_fallback(
        self,
        predictions: list[str],
        references: list[str],
    ) -> tuple[float, float, float]:
        """Simple token-overlap ROUGE-1 fallback (no external deps)."""
        def tokenize(text: str) -> list[str]:
            return text.lower().split()

        def ngrams(tokens: list[str], n: int) -> set:
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        def f1(pred_ng: set, ref_ng: set) -> float:
            if not pred_ng or not ref_ng:
                return 0.0
            overlap = len(pred_ng & ref_ng)
            p = overlap / len(pred_ng)
            r = overlap / len(ref_ng)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        r1s, r2s, rLs = [], [], []
        for pred, ref in zip(predictions, references):
            pt, rt = tokenize(pred), tokenize(ref)
            r1s.append(f1(ngrams(pt, 1), ngrams(rt, 1)))
            r2s.append(f1(ngrams(pt, 2), ngrams(rt, 2)))
            # ROUGE-L: LCS-based F1 (simplified)
            lcs = self._lcs_length(pt, rt)
            p = lcs / len(pt) if pt else 0
            r = lcs / len(rt) if rt else 0
            rLs.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

        return (
            sum(r1s) / len(r1s),
            sum(r2s) / len(r2s),
            sum(rLs) / len(rLs),
        )

    def _lcs_length(self, a: list, b: list) -> int:
        """LCS length for ROUGE-L fallback."""
        m, n = len(a), len(b)
        # Space-efficient LCS
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                curr[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(curr[j-1], prev[j])
            prev = curr
        return prev[n]

    def _compute_bert_score(
        self,
        predictions: list[str],
        references: list[str],
    ) -> Optional[float]:
        try:
            from bert_score import score as bert_score
            P, R, F = bert_score(predictions, references, lang="en", verbose=False)
            return float(F.mean())
        except ImportError:
            return None

    def _compute_rmse(self, predictions: list[float], references: list[float]) -> float:
        mse = sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)
        return math.sqrt(mse)

    # -----------------------------------------------------------------------
    # Task B
    # -----------------------------------------------------------------------

    def evaluate_task_b(
        self,
        ranked_lists: list[list[str]],
        relevant_items: list[list[str]],
        k_values: list[int] = None,
    ) -> TaskBResults:
        """
        ranked_lists:   list of ranked item_id lists (one per user), best first
        relevant_items: list of ground-truth relevant item_id sets (one per user)
        k_values:       list of K values to evaluate at (default [5, 10])
        """
        if k_values is None:
            k_values = [5, 10]

        assert len(ranked_lists) == len(relevant_items), "Lengths must match"

        ndcg_scores = {k: [] for k in k_values}
        hit_rates   = {k: [] for k in k_values}
        rr_scores   = []

        for ranked, relevant in zip(ranked_lists, relevant_items):
            relevant_set = set(relevant)

            # NDCG@K
            for k in k_values:
                ndcg_scores[k].append(self._ndcg(ranked, relevant_set, k))
                hit_rates[k].append(1.0 if any(r in relevant_set for r in ranked[:k]) else 0.0)

            # MRR
            rr = 0.0
            for i, item in enumerate(ranked, 1):
                if item in relevant_set:
                    rr = 1.0 / i
                    break
            rr_scores.append(rr)

        return TaskBResults(
            ndcg_at_k={k: sum(v) / len(v) for k, v in ndcg_scores.items()},
            hit_rate_at_k={k: sum(v) / len(v) for k, v in hit_rates.items()},
            mrr=sum(rr_scores) / len(rr_scores),
            num_users=len(ranked_lists),
        )

    def _ndcg(self, ranked: list[str], relevant: set, k: int) -> float:
        """Compute NDCG@K for a single user."""
        def dcg(items: list[str], rel: set, k: int) -> float:
            score = 0.0
            for i, item in enumerate(items[:k], 1):
                if item in rel:
                    score += 1.0 / math.log2(i + 1)
            return score

        actual_dcg = dcg(ranked, relevant, k)
        # Ideal: all relevant items at the top
        ideal_ranked = list(relevant)[:k]
        ideal_dcg = dcg(ideal_ranked, relevant, k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
