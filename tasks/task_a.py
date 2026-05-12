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
    attributes: list = None
    price_range: str = ""
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
    predicted_rating: float
    review_text: str
    confidence: str
    reasoning: str


# ---------------------------------------------------------------------------
# Review simulation agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at user behaviour modelling and persona simulation.
Your task is to simulate how a specific user would review a product or service they have
never seen before, based on everything known about their reviewing patterns and preferences.

The simulation must:
1. Predict a star rating consistent with the user's rating tendencies
2. Write a review that matches their vocabulary, tone, length, and style EXACTLY
3. Mirror the sentence structure and phrasing from their actual review examples
4. Reference the specific attributes of the item that align with or contradict their preferences
5. Use Nigerian Pidgin English and cultural references if the user's profile indicates this
6. Sound authentic — like this exact person wrote it, not a generic review

You must respond ONLY with a JSON object — no preamble, no markdown fences."""


def simulate_review(
    profile: UserProfile,
    item: ItemDetails,
    client: Optional[anthropic.Anthropic] = None,
    few_shot_reviews: list = None,
) -> SimulatedReview:
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    few_shot_block = ""
    if few_shot_reviews:
        examples = few_shot_reviews[-8:]
        few_shot_block = "\n\nEXAMPLES OF THIS USER'S ACTUAL REVIEWS (mirror their style exactly):\n"
        for r in examples:
            few_shot_block += (
                f"\n[{r.get('category', '')} | {r.get('rating', '?')}/5 stars]\n"
                f"{r.get('text', '')}\n"
            )

    prompt = f"""You must simulate how this specific user would review the item below.
Your goal is to sound EXACTLY like this person — same vocabulary, same sentence structure,
same length, same quirks. A judge should not be able to tell the difference.

{profile.to_prompt_context()}
{few_shot_block}

ITEM TO REVIEW:
{item.to_prompt_text()}

STRICT STYLE RULES — follow these exactly:
- Match their typical review length of ~{profile.style.avg_length_words} words
- Use their tone: {profile.style.tone} and formality: {profile.style.formality}
- Mirror their sentence structure from the examples above
- Reuse their characteristic phrases where natural: {', '.join(profile.style.signature_phrases[:5]) if profile.style.signature_phrases else 'none'}
- Uses Pidgin: {'YES — use Nigerian Pidgin English naturally throughout' if profile.style.uses_pidgin else 'No Pidgin'}
- Punctuation style: {profile.style.punctuation_style}

RATING LOGIC:
- Their average is {profile.rating.mean:.1f}/5, tendency: {profile.rating.tendency}
- Loved attributes to praise if present: {', '.join(profile.preferences.loved_attributes[:4])}
- Disliked attributes to criticise if present: {', '.join(profile.preferences.disliked_attributes[:4])}
- Deal-breakers that drop rating: {', '.join(profile.preferences.deal_breakers[:3])}
- Must-haves that raise rating: {', '.join(profile.preferences.must_haves[:3])}

Reason through:
1. Which item attributes match or conflict with their preferences?
2. What rating (to nearest 0.5) would they give?
3. Write the review in their exact voice — not a generic review.

Return ONLY this JSON:
{{
  "predicted_rating": <float 1.0-5.0, nearest 0.5>,
  "review_text": "<the simulated review — must sound like this specific person>",
  "confidence": "<high|medium|low>",
  "reasoning": "<2 sentences on rating and style choices>"
}}"""

    raw = ""
    for attempt in range(3):
        try:
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

            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)

            raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
            raw = re.sub(r',\s*([}\]])', r'\1', raw)

            data = json.loads(raw.strip())
            break

        except json.JSONDecodeError:
            if attempt < 2:
                import time
                time.sleep(2)
                continue
            else:
                raise

    return SimulatedReview(
        user_id=profile.user_id,
        item_id=item.item_id,
        predicted_rating=float(data["predicted_rating"]),
        review_text=data["review_text"],
        confidence=data["confidence"],
        reasoning=data["reasoning"],
    )


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def batch_simulate(
    profile: UserProfile,
    items: list,
    few_shot_reviews: list = None,
    client: Optional[anthropic.Anthropic] = None,
) -> list:
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    results = []
    for item in items:
        result = simulate_review(profile, item, client=client, few_shot_reviews=few_shot_reviews)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
        from tasks.frontend import HTML
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(
        title="DSN BCT Task A — User Review Simulation",
        description="Simulates user reviews and ratings for unseen items based on user history.",
        version="1.0.0",
    )

    class ReviewRequest(BaseModel):
        user_id: str
        review_history: list
        item: dict
        include_reasoning: bool = False

    class ReviewResponse(BaseModel):
        user_id: str
        item_id: str
        predicted_rating: float
        review_text: str
        confidence: str
        reasoning: str = ""

    _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @app.get("/", response_class=HTMLResponse)
    async def homepage():
        return HTML

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