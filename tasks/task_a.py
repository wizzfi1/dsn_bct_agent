"""
Task A: User Modeling Agent
============================
Given a UserProfile and an item the user has NOT reviewed,
simulate:
  1. The star rating they would give (1–5)
  2. A written review in their voice

Evaluated on:
  - Review text quality   (ROUGE / BERTScore)
  - Rating accuracy       (RMSE)
  - Behavioural fidelity  (human eval — does it sound like them?)
  - Nigerian localisation bonus

API contract (for containerised submission):
  Input:  UserPersona (user_id + review history OR pre-built profile)
          ItemDetails (name, category, attributes, description)
  Output: SimulatedReview (rating, text, confidence, reasoning)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic
from core.user_profile import UserProfile, build_user_profile


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ItemDetails:
    item_id: str
    item_name: str
    category: str
    description: str = ""
    attributes: list[str] = None          # e.g. ["fast service", "outdoor seating"]
    price_range: str = ""                 # e.g. "$$" or "affordable" or "premium"
    location: str = ""

    def to_prompt_text(self) -> str:
        parts = [
            f"Item: {self.item_name}",
            f"Category: {self.category}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.attributes:
            parts.append(f"Attributes: {', '.join(self.attributes)}")
        if self.price_range:
            parts.append(f"Price range: {self.price_range}")
        if self.location:
            parts.append(f"Location: {self.location}")
        return "\n".join(parts)


@dataclass
class SimulatedReview:
    user_id: str
    item_id: str
    predicted_rating: float              # 1.0 – 5.0
    review_text: str
    confidence: str                      # "high" | "medium" | "low"
    reasoning: str                       # chain-of-thought (for paper/debugging)


# ---------------------------------------------------------------------------
# Review simulation agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at user behaviour modelling and persona simulation.
Your task is to simulate how a specific user would review a product or service they have
never seen before, based on everything known about their reviewing patterns and preferences.

The simulation must:
1. Predict a star rating consistent with the user's rating tendencies
2. Write a review that matches their vocabulary, tone, length, and style
3. Reference the specific attributes of the item that align with or contradict their preferences
4. Use Nigerian Pidgin English and cultural references if the user's profile indicates this
5. Sound authentic — like this exact person wrote it, not a generic review

You must respond ONLY with a JSON object — no preamble, no markdown fences."""


def simulate_review(
    profile: UserProfile,
    item: ItemDetails,
    client: Optional[anthropic.Anthropic] = None,
    few_shot_reviews: list[dict] = None,
) -> SimulatedReview:
    """
    Simulate a user's review for an item they haven't reviewed.

    profile: pre-built UserProfile from the User Profile Engine
    item: details about the item to review
    few_shot_reviews: optional list of the user's actual past reviews
                      (most powerful signal — include if available)
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Build few-shot examples block
    few_shot_block = ""
    if few_shot_reviews:
        examples = few_shot_reviews[-5:]  # most recent 5
        few_shot_block = "\n\nEXAMPLES OF THIS USER'S ACTUAL REVIEWS (use as style reference):\n"
        for r in examples:
            few_shot_block += (
                f"\n[{r.get('category', '')} | {r.get('rating', '?')}/5 stars]\n"
                f"{r.get('text', '')}\n"
            )

    prompt = f"""You must simulate how this specific user would review the item below.

{profile.to_prompt_context()}
{few_shot_block}

ITEM TO REVIEW:
{item.to_prompt_text()}

SIMULATION INSTRUCTIONS:
- Their average rating is {profile.rating.mean:.1f}/5 with tendency: {profile.rating.tendency}
- They typically write {profile.style.length_tendency} reviews (~{profile.style.avg_length_words} words)
- Their tone is {profile.style.tone}, formality: {profile.style.formality}
- Uses Pidgin: {'YES — incorporate naturally' if profile.style.uses_pidgin else 'No'}
- Loved attributes to praise if present: {', '.join(profile.preferences.loved_attributes[:4])}
- Disliked attributes to criticise if present: {', '.join(profile.preferences.disliked_attributes[:4])}
- Deal-breakers (would drop rating significantly): {', '.join(profile.preferences.deal_breakers[:3])}

Reason through:
1. Which item attributes match or conflict with their preferences?
2. What rating would this person most likely give and why?
3. How would they express this in their natural voice?

Return ONLY this JSON:
{{
  "predicted_rating": <float 1.0-5.0, round to nearest 0.5>,
  "review_text": "<the simulated review>",
  "confidence": "<high|medium|low>",
  "reasoning": "<2-3 sentences explaining your rating and style choices>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw.strip())

    return SimulatedReview(
        user_id=profile.user_id,
        item_id=item.item_id,
        predicted_rating=float(data["predicted_rating"]),
        review_text=data["review_text"],
        confidence=data["confidence"],
        reasoning=data["reasoning"],
    )


# ---------------------------------------------------------------------------
# Batch simulation (for evaluation runs)
# ---------------------------------------------------------------------------

def batch_simulate(
    profile: UserProfile,
    items: list[ItemDetails],
    few_shot_reviews: list[dict] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> list[SimulatedReview]:
    """Run simulation for multiple items for a single user."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    results = []
    for item in items:
        result = simulate_review(profile, item, client=client, few_shot_reviews=few_shot_reviews)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# FastAPI app (containerised submission endpoint)
# ---------------------------------------------------------------------------

def create_app():
    """
    Creates the FastAPI application for Task A submission.
    Run with: uvicorn tasks.task_a:create_app --factory --host 0.0.0.0 --port 8000
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(
        title="DSN BCT Task A — User Review Simulation",
        description="Simulates user reviews and ratings for unseen items based on user history.",
        version="1.0.0",
    )

    class ReviewRequest(BaseModel):
        user_id: str
        review_history: list[dict]       # list of past review dicts
        item: dict                       # ItemDetails as dict
        include_reasoning: bool = False

    class ReviewResponse(BaseModel):
        user_id: str
        item_id: str
        predicted_rating: float
        review_text: str
        confidence: str
        reasoning: str = ""

    _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @app.post("/simulate", response_model=ReviewResponse)
    async def simulate_endpoint(request: ReviewRequest):
        try:
            if not request.review_history:
                raise HTTPException(status_code=400, detail="review_history cannot be empty")

            profile = build_user_profile(
                user_id=request.user_id,
                reviews=request.review_history,
                client=_client,
            )

            item_data = request.item
            item = ItemDetails(
                item_id=item_data.get("item_id", "unknown"),
                item_name=item_data.get("item_name", ""),
                category=item_data.get("category", ""),
                description=item_data.get("description", ""),
                attributes=item_data.get("attributes", []),
                price_range=item_data.get("price_range", ""),
                location=item_data.get("location", ""),
            )

            result = simulate_review(
                profile=profile,
                item=item,
                client=_client,
                few_shot_reviews=request.review_history,
            )

            return ReviewResponse(
                user_id=result.user_id,
                item_id=result.item_id,
                predicted_rating=result.predicted_rating,
                review_text=result.review_text,
                confidence=result.confidence,
                reasoning=result.reasoning if request.include_reasoning else "",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "ok", "task": "A"}

    return app
