"""
Task B: Recommendation Agent
==============================
Given a UserProfile, rank and recommend items tailored to that user.

Handles:
  - Warm users    (have review history → use UserProfile)
  - Cold-start    (new users → elicit preferences via 2-3 questions)
  - Cross-domain  (recommend across different categories)
  - Multi-turn    (update profile as conversation progresses)

Agentic workflow (reason before recommending):
  1. Analyse user profile  →  identify preference vectors
  2. Retrieve candidates   →  filter item pool by compatibility
  3. Score & rank          →  score each candidate against profile
  4. Explain               →  generate natural language justification

Evaluated on:
  - NDCG@10 / Hit Rate      (30 pts)
  - Cold-start & cross-domain performance (25 pts)
  - Contextual relevance / human eval     (20 pts)
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import anthropic
from core.user_profile import UserProfile, build_user_profile


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateItem:
    item_id: str
    item_name: str
    category: str
    description: str = ""
    attributes: list[str] = None
    price_range: str = ""
    avg_rating: float = 0.0
    popularity: int = 0               # number of reviews (proxy for popularity)


@dataclass
class RankedRecommendation:
    item: CandidateItem
    score: float                      # 0.0 – 1.0 relevance score
    rank: int
    explanation: str                  # why this was recommended
    matched_preferences: list[str]    # which user preferences this satisfies
    caveats: list[str]                # potential mismatches to warn about


@dataclass
class RecommendationResult:
    user_id: str
    recommendations: list[RankedRecommendation]
    reasoning_trace: str              # full agent reasoning (for paper)
    cold_start: bool                  # True if profile was built from elicitation


# ---------------------------------------------------------------------------
# Cold-start elicitation
# ---------------------------------------------------------------------------

COLD_START_QUESTIONS = [
    {
        "id": "q1",
        "question": "What types of things do you usually enjoy? (e.g. food, books, electronics, restaurants)",
        "type": "open",
    },
    {
        "id": "q2",
        "question": "On a scale of 1–5, how would you typically rate something you find just okay (not great, not terrible)?",
        "type": "rating_anchor",
    },
    {
        "id": "q3",
        "question": "What's one thing that would immediately make you give a low rating to anything?",
        "type": "deal_breaker",
    },
]


def build_cold_start_profile(
    user_id: str,
    elicitation_answers: dict,
    client: Optional[anthropic.Anthropic] = None,
) -> UserProfile:
    """
    Build a UserProfile for a new user from structured elicitation answers.
    elicitation_answers: {"q1": "...", "q2": "3", "q3": "..."}
    """
    from core.user_profile import (
        UserProfile, RatingProfile, StyleProfile,
        PreferenceProfile, BehaviouralProfile
    )

    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    q1 = elicitation_answers.get("q1", "")
    q2 = elicitation_answers.get("q2", "3")
    q3 = elicitation_answers.get("q3", "")

    # Use rating anchor to estimate their tendency
    try:
        anchor = float(q2)
    except ValueError:
        anchor = 3.0

    # LLM interprets qualitative answers into structured preferences
    prompt = f"""A new user answered three preference questions. Extract their profile.

Q1 (interests): {q1}
Q2 (rating anchor — they'd rate an "okay" item): {q2}/5
Q3 (deal-breaker): {q3}

Return ONLY this JSON:
{{
  "top_categories": ["<cat>", ...],
  "loved_attributes": ["<attr>", ...],
  "disliked_attributes": ["<attr>", ...],
  "deal_breakers": ["<attr>"],
  "must_haves": ["<attr>", ...],
  "tone": "<warm|neutral|critical|enthusiastic>",
  "summary": "<1-2 sentence profile>"
}}"""

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())

    # Estimate mean rating: if they'd give 3 to "okay", they're balanced
    mean = 3.0 + (anchor - 3.0) * 0.5
    tendency = "generous" if mean >= 4.0 else "harsh" if mean <= 2.5 else "balanced"

    return UserProfile(
        user_id=user_id,
        num_reviews=0,
        rating=RatingProfile(
            mean=round(mean, 1),
            std=1.0,
            distribution={},
            tendency=tendency,
            context_sensitivity="unknown",
        ),
        style=StyleProfile(
            avg_length_words=80,
            length_tendency="moderate",
            formality="mixed",
            tone=data.get("tone", "neutral"),
            uses_pidgin=False,
            signature_phrases=[],
            punctuation_style="standard",
        ),
        preferences=PreferenceProfile(
            top_categories=data.get("top_categories", []),
            loved_attributes=data.get("loved_attributes", []),
            disliked_attributes=data.get("disliked_attributes", []),
            deal_breakers=data.get("deal_breakers", []),
            must_haves=data.get("must_haves", []),
            cross_domain_interests=[],
        ),
        behaviour=BehaviouralProfile(
            review_frequency="occasional",
            complain_ratio=0.2,
            detail_orientation="surface",
            social_signals=[],
            cultural_context=[],
        ),
        raw_summary=data.get("summary", "New user with limited history."),
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Recommendation agent
# ---------------------------------------------------------------------------

RECOMMENDATION_SYSTEM = """You are an expert recommendation agent. Your job is to rank
candidate items for a specific user based on their profile, and explain your reasoning clearly.

You think step-by-step:
1. What does this user care about most?
2. Which candidates match those priorities?
3. Are there any deal-breakers in the candidates?
4. What is the final ranked order with scores?

Nigerian cultural context matters: if the user has Nigerian cultural signals,
factor in local relevance, familiarity, and cultural fit.

Respond ONLY with a JSON object — no preamble, no markdown fences."""


def recommend(
    profile: UserProfile,
    candidates: list[CandidateItem],
    top_k: int = 10,
    context: str = "",
    client: Optional[anthropic.Anthropic] = None,
) -> RecommendationResult:
    """
    Core recommendation function.

    profile: UserProfile (warm or cold-start)
    candidates: pool of items to rank
    top_k: number of recommendations to return (NDCG@10 requires k>=10)
    context: optional additional context ("looking for a birthday dinner", etc.)
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Format candidate pool
    candidates_block = "\n".join(
        f"[{i+1}] id={c.item_id} | {c.item_name} | {c.category} | "
        f"avg_rating={c.avg_rating:.1f} | "
        f"attrs: {', '.join(c.attributes or [])} | "
        f"price: {c.price_range} | "
        f"desc: {c.description[:100]}"
        for i, c in enumerate(candidates)
    )

    context_block = f"\nADDITIONAL CONTEXT FROM USER: {context}\n" if context else ""

    prompt = f"""{profile.to_prompt_context()}
{context_block}
CANDIDATE ITEMS TO RANK:
{candidates_block}

TASK: Rank the top {top_k} items for this user.

For each recommended item, reason through:
- Does it match their loved attributes?
- Does it trigger any deal-breakers?
- Is it relevant to their top categories?
- How does the price range fit their expectations?

Return ONLY this JSON:
{{
  "reasoning_trace": "<your step-by-step thinking process>",
  "recommendations": [
    {{
      "item_id": "<id>",
      "score": <float 0.0-1.0>,
      "explanation": "<why this is recommended for this specific user>",
      "matched_preferences": ["<pref1>", ...],
      "caveats": ["<potential mismatch>", ...]
    }},
    ...
  ]
}}

Order recommendations from best match (score closest to 1.0) to worst.
Include exactly {top_k} items."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=RECOMMENDATION_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())

    # Map item_id back to CandidateItem objects
    item_map = {c.item_id: c for c in candidates}
    ranked = []
    for rank, r in enumerate(data["recommendations"], 1):
        item = item_map.get(r["item_id"])
        if item is None:
            continue
        ranked.append(RankedRecommendation(
            item=item,
            score=float(r["score"]),
            rank=rank,
            explanation=r["explanation"],
            matched_preferences=r.get("matched_preferences", []),
            caveats=r.get("caveats", []),
        ))

    return RecommendationResult(
        user_id=profile.user_id,
        recommendations=ranked,
        reasoning_trace=data.get("reasoning_trace", ""),
        cold_start=(profile.num_reviews == 0),
    )


# ---------------------------------------------------------------------------
# Multi-turn conversation handler
# ---------------------------------------------------------------------------

class RecommendationSession:
    """
    Manages a multi-turn recommendation conversation.
    The agent updates its understanding of the user as they give feedback.
    """

    def __init__(self, profile: UserProfile, candidates: list[CandidateItem]):
        self.profile = profile
        self.candidates = candidates
        self.conversation_history: list[dict] = []
        self.excluded_items: set[str] = set()
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def get_recommendations(self, user_message: str = "", top_k: int = 5) -> RecommendationResult:
        """Get or refine recommendations based on user feedback."""

        # Exclude already-rejected items
        active_candidates = [c for c in self.candidates if c.item_id not in self.excluded_items]

        context = " | ".join(
            f"{m['role']}: {m['content']}"
            for m in self.conversation_history[-4:]  # last 2 turns
        )
        if user_message:
            context += f" | Latest feedback: {user_message}"

        result = recommend(
            profile=self.profile,
            candidates=active_candidates,
            top_k=top_k,
            context=context,
            client=self.client,
        )

        if user_message:
            self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Recommended: {[r.item.item_name for r in result.recommendations[:3]]}",
        })

        return result

    def reject_item(self, item_id: str):
        """User explicitly rejects an item — exclude from future recommendations."""
        self.excluded_items.add(item_id)


# ---------------------------------------------------------------------------
# FastAPI app (containerised submission endpoint)
# ---------------------------------------------------------------------------

def create_app():
    """
    Creates the FastAPI application for Task B submission.
    Run with: uvicorn tasks.task_b:create_app --factory --host 0.0.0.0 --port 8001
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(
        title="DSN BCT Task B — Personalised Recommendation",
        description="Delivers personalised recommendations from a user persona.",
        version="1.0.0",
    )

    class WarmRequest(BaseModel):
        user_id: str
        review_history: list[dict]
        candidates: list[dict]
        top_k: int = 10
        context: str = ""

    class ColdStartRequest(BaseModel):
        user_id: str
        elicitation_answers: dict     # {"q1": "...", "q2": "3", "q3": "..."}
        candidates: list[dict]
        top_k: int = 10

    class RecResponse(BaseModel):
        user_id: str
        recommendations: list[dict]
        cold_start: bool

    _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _parse_candidates(raw: list[dict]) -> list[CandidateItem]:
        return [
            CandidateItem(
                item_id=c.get("item_id", ""),
                item_name=c.get("item_name", ""),
                category=c.get("category", ""),
                description=c.get("description", ""),
                attributes=c.get("attributes", []),
                price_range=c.get("price_range", ""),
                avg_rating=float(c.get("avg_rating", 0)),
                popularity=int(c.get("popularity", 0)),
            )
            for c in raw
        ]

    @app.post("/recommend/warm", response_model=RecResponse)
    async def recommend_warm(request: WarmRequest):
        try:
            profile = build_user_profile(request.user_id, request.review_history, _client)
            candidates = _parse_candidates(request.candidates)
            result = recommend(profile, candidates, request.top_k, request.context, _client)
            return RecResponse(
                user_id=result.user_id,
                recommendations=[
                    {
                        "rank": r.rank,
                        "item_id": r.item.item_id,
                        "item_name": r.item.item_name,
                        "score": r.score,
                        "explanation": r.explanation,
                        "matched_preferences": r.matched_preferences,
                        "caveats": r.caveats,
                    }
                    for r in result.recommendations
                ],
                cold_start=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/recommend/cold-start", response_model=RecResponse)
    async def recommend_cold(request: ColdStartRequest):
        try:
            profile = build_cold_start_profile(request.user_id, request.elicitation_answers, _client)
            candidates = _parse_candidates(request.candidates)
            result = recommend(profile, candidates, request.top_k, client=_client)
            return RecResponse(
                user_id=result.user_id,
                recommendations=[
                    {
                        "rank": r.rank,
                        "item_id": r.item.item_id,
                        "item_name": r.item.item_name,
                        "score": r.score,
                        "explanation": r.explanation,
                        "matched_preferences": r.matched_preferences,
                        "caveats": r.caveats,
                    }
                    for r in result.recommendations
                ],
                cold_start=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/cold-start/questions")
    async def get_cold_start_questions():
        return {"questions": COLD_START_QUESTIONS}

    @app.get("/health")
    async def health():
        return {"status": "ok", "task": "B"}

    return app
