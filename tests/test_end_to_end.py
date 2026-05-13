"""
End-to-end smoke test

Runs both Task A and Task B using the sample Nigerian reviewer
from data/loaders.py — no real dataset needed.

Run with: python tests/test_end_to_end.py
Requires: ANTHROPIC_API_KEY environment variable set
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_profile import build_user_profile
from data.loaders import SAMPLE_REVIEWS
from tasks.task_a import simulate_review, ItemDetails
from tasks.task_b import recommend, CandidateItem, build_cold_start_profile
from evaluation.metrics import Evaluator


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_profile_engine():
    print_section("1. USER PROFILE ENGINE")
    profile = build_user_profile(
        user_id="user_ng_001",
        reviews=SAMPLE_REVIEWS,
    )
    print(f"\nProfile built for {profile.user_id}")
    print(f"Reviews analysed: {profile.num_reviews}")
    print(f"Confidence: {profile.confidence}")
    print(f"\n{profile.to_prompt_context()}")
    return profile


def test_task_a(profile):
    print_section("2. TASK A — REVIEW SIMULATION")

    # Simulate a review for a Nigerian restaurant the user hasn't visited
    new_item = ItemDetails(
        item_id="biz_new_001",
        item_name="Iya Basira Restaurant",
        category="Nigerian Cuisine",
        description="Authentic local buka serving traditional Nigerian dishes",
        attributes=["affordable", "local food", "pepper soup", "eba and soup", "outdoor seating"],
        price_range="₦",
        location="Lagos Island",
    )

    result = simulate_review(
        profile=profile,
        item=new_item,
        few_shot_reviews=SAMPLE_REVIEWS,
    )

    print(f"\nItem: {new_item.item_name}")
    print(f"Predicted rating: {result.predicted_rating}/5")
    print(f"Confidence: {result.confidence}")
    print(f"\nSimulated review:\n{result.review_text}")
    print(f"\nReasoning: {result.reasoning}")
    return result


def test_task_a_mismatch(profile):
    print_section("2b. TASK A — MISMATCH ITEM (predicting a negative review)")

    # Item likely to trigger dislikes (poor service, overpriced)
    bad_fit = ItemDetails(
        item_id="biz_new_002",
        item_name="Fancy Rooftop Lounge",
        category="Bars & Nightlife",
        description="Upscale rooftop bar with imported cocktails and minimal Nigerian food options",
        attributes=["expensive", "slow service", "limited menu", "no local food"],
        price_range="₦₦₦₦",
    )

    result = simulate_review(
        profile=profile,
        item=bad_fit,
        few_shot_reviews=SAMPLE_REVIEWS,
    )

    print(f"\nItem: {bad_fit.item_name}")
    print(f"Predicted rating: {result.predicted_rating}/5")
    print(f"\nSimulated review:\n{result.review_text}")


def test_task_b_warm(profile):
    print_section("3. TASK B — RECOMMENDATION (warm user)")

    candidates = [
        CandidateItem("c001", "Suya Spot Allen", "Nigerian Cuisine",
                      "Famous for grilled suya and kilishi", ["suya", "grilled meat", "outdoor"], "₦", 4.7, 850),
        CandidateItem("c002", "Chinese Palace", "Chinese Restaurant",
                      "Authentic Chinese dim sum", ["dim sum", "noodles", "AC", "quiet"], "₦₦₦", 4.2, 300),
        CandidateItem("c003", "Tantalizers Surulere", "Fast Food",
                      "Nigerian fast food chain", ["fast service", "local menu", "affordable"], "₦", 3.8, 1200),
        CandidateItem("c004", "The Yellow Chilli", "Nigerian Fine Dining",
                      "Upscale Nigerian cuisine by celebrity chef", ["premium", "local dishes", "elegant"], "₦₦₦₦", 4.8, 500),
        CandidateItem("c005", "Pizza Palace", "Pizza",
                      "American-style pizza delivery", ["delivery", "pepperoni", "cheese"], "₦₦", 3.5, 600),
        CandidateItem("c006", "Amala Joint", "Nigerian Cuisine",
                      "Specialist in amala, gbegiri, and ewedu", ["amala", "traditional", "local", "affordable"], "₦", 4.5, 950),
        CandidateItem("c007", "Barcelos", "Grills",
                      "Portuguese-inspired peri-peri grills", ["grilled chicken", "fast", "medium price"], "₦₦", 4.1, 700),
        CandidateItem("c008", "The Grill House", "Continental",
                      "Western continental food", ["steak", "burgers", "expensive", "quiet"], "₦₦₦", 4.0, 250),
        CandidateItem("c009", "Mama Put Central", "Street Food",
                      "Classic roadside mama put with rotating daily menu", ["affordable", "local", "rice dishes"], "₦", 4.3, 2000),
        CandidateItem("c010", "Sky Lounge Bar", "Bars",
                      "Upscale bar with cocktails and limited food", ["drinks", "expensive", "slow service"], "₦₦₦₦", 3.9, 200),
    ]

    result = recommend(
        profile=profile,
        candidates=candidates,
        top_k=5,
        context="Looking for a good dinner spot for the weekend",
    )

    print(f"\nTop 5 recommendations for {profile.user_id}:")
    for r in result.recommendations:
        print(f"\n  #{r.rank} {r.item.item_name} — score: {r.score:.2f}")
        print(f"       {r.explanation}")
        if r.matched_preferences:
            print(f"       Matches: {', '.join(r.matched_preferences)}")
        if r.caveats:
            print(f"       Caveats: {', '.join(r.caveats)}")

    print(f"\nAgent reasoning trace:\n{result.reasoning_trace[:400]}...")
    return result


def test_task_b_cold_start():
    print_section("4. TASK B — COLD-START (new user)")

    # Simulate a brand new user answering 3 elicitation questions
    answers = {
        "q1": "I love Nigerian food especially rice dishes and grilled meat. Also enjoy trying new restaurants.",
        "q2": "3",
        "q3": "Terrible service and overpriced food with no value for money",
    }

    cold_profile = build_cold_start_profile(
        user_id="new_user_999",
        elicitation_answers=answers,
    )

    print(f"\nCold-start profile built:")
    print(f"  Top categories: {cold_profile.preferences.top_categories}")
    print(f"  Loved attributes: {cold_profile.preferences.loved_attributes}")
    print(f"  Deal-breakers: {cold_profile.preferences.deal_breakers}")
    print(f"  Confidence: {cold_profile.confidence}")

    # Reuse same candidates
    candidates = [
        CandidateItem("c001", "Suya Spot Allen", "Nigerian Cuisine", "", ["suya", "affordable"], "₦", 4.7, 850),
        CandidateItem("c002", "Chinese Palace", "Chinese Restaurant", "", ["quiet", "expensive"], "₦₦₦", 4.2, 300),
        CandidateItem("c006", "Amala Joint", "Nigerian Cuisine", "", ["amala", "local", "affordable"], "₦", 4.5, 950),
        CandidateItem("c009", "Mama Put Central", "Street Food", "", ["affordable", "local"], "₦", 4.3, 2000),
        CandidateItem("c010", "Sky Lounge Bar", "Bars", "", ["expensive", "slow service"], "₦₦₦₦", 3.9, 200),
    ]

    result = recommend(cold_profile, candidates, top_k=3)

    print(f"\nCold-start recommendations:")
    for r in result.recommendations:
        print(f"  #{r.rank} {r.item.item_name} — score: {r.score:.2f} | {r.explanation}")


def test_evaluation():
    print_section("5. EVALUATION METRICS")

    evaluator = Evaluator()

    # Task A eval (dummy data)
    predictions_a = [
        {"text": "The food was great and the service was fast. I loved the jollof rice.", "rating": 4.0},
        {"text": "Very disappointing. Cold food and rude staff. Will not return.", "rating": 2.0},
    ]
    ground_truth_a = [
        {"text": "Excellent food! The rice was perfectly cooked and staff were friendly.", "rating": 4.5},
        {"text": "Bad experience. The meal was cold and the waiter was rude.", "rating": 1.5},
    ]
    results_a = evaluator.evaluate_task_a(predictions_a, ground_truth_a)
    print(f"\n{results_a.summary()}")

    # Task B eval (dummy data)
    ranked_lists = [
        ["c001", "c006", "c009", "c002", "c010"],
        ["c006", "c001", "c009", "c010", "c002"],
    ]
    relevant_items = [
        ["c001", "c006"],
        ["c006", "c009"],
    ]
    results_b = evaluator.evaluate_task_b(ranked_lists, relevant_items, k_values=[3, 5])
    print(f"\n{results_b.summary([3, 5])}")


if __name__ == "__main__":
    print("\nDSN BCT Hackathon — End-to-end test")
    print("Requires ANTHROPIC_API_KEY to be set\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Export it with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    profile = test_profile_engine()
    test_task_a(profile)
    test_task_a_mismatch(profile)
    test_task_b_warm(profile)
    test_task_b_cold_start()
    test_evaluation()

    print("\n\nAll tests complete!")
