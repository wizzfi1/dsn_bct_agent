"""
Task B: Recommendation Agent

Given a UserProfile, rank and recommend items tailored to that user.

Handles:
  - Warm users    (have review history → use UserProfile)
  - Cold-start    (new users → elicit preferences via 2-3 questions)
  - Cross-domain  (recommend across different categories)
  - Multi-turn    (update profile as conversation progresses)

Evaluated on:
  - NDCG@10 / Hit Rate      (30 pts)
  - Cold-start & cross-domain performance (25 pts)
  - Contextual relevance / human eval     (20 pts)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import anthropic
from core.user_profile import UserProfile, build_user_profile


@dataclass
class CandidateItem:
    item_id: str
    item_name: str
    category: str
    description: str = ""
    attributes: list = None
    price_range: str = ""
    avg_rating: float = 0.0
    popularity: int = 0


@dataclass
class RankedRecommendation:
    item: CandidateItem
    score: float
    rank: int
    explanation: str
    matched_preferences: list
    caveats: list


@dataclass
class RecommendationResult:
    user_id: str
    recommendations: list
    reasoning_trace: str
    cold_start: bool


COLD_START_QUESTIONS = [
    {
        "id": "q1",
        "question": "What types of things do you usually enjoy? (e.g. food, books, electronics, restaurants)",
        "type": "open",
    },
    {
        "id": "q2",
        "question": "On a scale of 1-5, how would you typically rate something you find just okay (not great, not terrible)?",
        "type": "rating_anchor",
    },
    {
        "id": "q3",
        "question": "What's one thing that would immediately make you give a low rating to anything?",
        "type": "deal_breaker",
    },
]


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw.strip()


def build_cold_start_profile(
    user_id: str,
    elicitation_answers: dict,
    client: Optional[anthropic.Anthropic] = None,
) -> UserProfile:
    from core.user_profile import (
        RatingProfile, StyleProfile,
        PreferenceProfile, BehaviouralProfile
    )

    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    q1 = elicitation_answers.get("q1", "")
    q2 = elicitation_answers.get("q2", "3")
    q3 = elicitation_answers.get("q3", "")

    try:
        anchor = float(q2)
    except ValueError:
        anchor = 3.0

    prompt = f"""A new user answered three preference questions. Extract their profile.

Q1 (interests): {q1}
Q2 (rating anchor): {q2}/5
Q3 (deal-breaker): {q3}

Return ONLY this JSON:
{{
  "top_categories": ["<cat1>", "<cat2>"],
  "loved_attributes": ["<attr1>", "<attr2>"],
  "disliked_attributes": ["<attr1>"],
  "deal_breakers": ["<attr1>"],
  "must_haves": ["<attr1>", "<attr2>"],
  "tone": "<warm|neutral|critical|enthusiastic>",
  "summary": "<1-2 sentence profile>"
}}"""

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _clean_json(resp.content[0].text)
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

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


RECOMMENDATION_SYSTEM = """You are an expert recommendation agent. Your job is to rank
candidate items for a specific user based on their profile, and explain your reasoning clearly.

You think step-by-step:
1. What does this user care about most?
2. Which candidates match those priorities?
3. Are there any deal-breakers in the candidates?
4. What is the final ranked order with scores?

Nigerian cultural context matters: if the user has Nigerian cultural signals,
factor in local relevance, familiarity, and cultural fit.

Respond ONLY with a JSON object — no preamble, no markdown fences.
Keep all string values concise — maximum 20 words per explanation."""


def recommend(
    profile: UserProfile,
    candidates: list,
    top_k: int = 10,
    context: str = "",
    client: Optional[anthropic.Anthropic] = None,
) -> RecommendationResult:
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    candidates = candidates[:25]
    top_k = min(top_k, len(candidates))

    candidates_block = "\n".join(
        f"[{i+1}] id={c.item_id} | {c.item_name[:40]} | {c.category} | "
        f"rating={c.avg_rating:.1f} | price={c.price_range}"
        for i, c in enumerate(candidates)
    )

    context_block = f"\nCONTEXT: {context}\n" if context else ""

    prompt = f"""{profile.to_prompt_context()}
{context_block}
CANDIDATES:
{candidates_block}

Rank the top {top_k} items for this user.

Return ONLY this JSON (keep explanations under 15 words each):
{{
  "reasoning_trace": "<brief step-by-step thinking, max 50 words>",
  "recommendations": [
    {{
      "item_id": "<exact id from candidates>",
      "score": <float 0.0-1.0>,
      "explanation": "<why recommended, max 15 words>",
      "matched_preferences": ["<pref1>"],
      "caveats": []
    }}
  ]
}}

Include exactly {top_k} recommendations ordered best to worst."""

    last_error = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=3000,
                system=RECOMMENDATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _clean_json(response.content[0].text)
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < 2:
                time.sleep(2)
                continue
            else:
                raise ValueError(f"JSON parse failed after 3 attempts: {last_error}")

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


class RecommendationSession:
    def __init__(self, profile: UserProfile, candidates: list):
        self.profile = profile
        self.candidates = candidates
        self.conversation_history: list = []
        self.excluded_items: set = set()
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def get_recommendations(self, user_message: str = "", top_k: int = 5) -> RecommendationResult:
        active_candidates = [c for c in self.candidates if c.item_id not in self.excluded_items]
        context = " | ".join(
            f"{m['role']}: {m['content']}"
            for m in self.conversation_history[-4:]
        )
        if user_message:
            context += f" | Latest: {user_message}"

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
        self.excluded_items.add(item_id)


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
        from tasks.frontend import HTML
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(
        title="DSN BCT Task B — Personalised Recommendation",
        description="Delivers personalised recommendations from a user persona.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class WarmRequest(BaseModel):
        user_id: str
        review_history: list
        candidates: list
        top_k: int = 10
        context: str = ""

    class ColdStartRequest(BaseModel):
        user_id: str
        elicitation_answers: dict
        candidates: list
        top_k: int = 10

    class RecResponse(BaseModel):
        user_id: str
        recommendations: list
        cold_start: bool

    _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _parse_candidates(raw: list) -> list:
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

    @app.get("/", response_class=HTMLResponse)
    async def homepage():
        return HTML

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