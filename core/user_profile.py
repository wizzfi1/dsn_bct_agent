"""
User Profile Engine
====================
Shared foundation for both Task A (User Modeling) and Task B (Recommendation).

Given a user's review history, this module extracts a rich, structured
UserProfile that captures:
  - Rating tendencies (generous, harsh, balanced)
  - Writing style (vocabulary, tone, length, formality)
  - Topic preferences (what they care about most)
  - Likes and dislikes (recurring themes)
  - Cultural/contextual signals (Nigerian localisation hooks)
  - Behavioural patterns (when they review, how often they complain)

Both Task A and Task B consume UserProfile as their primary input.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import anthropic

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RatingProfile:
    mean: float                      # average star rating given
    std: float                       # standard deviation (how variable they are)
    distribution: dict[str, int]     # {"1": n, "2": n, ... "5": n}
    tendency: str                    # "generous" | "harsh" | "balanced" | "bimodal"
    context_sensitivity: str         # "high" (rating varies a lot by context) | "low"


@dataclass
class StyleProfile:
    avg_length_words: int            # typical review length
    length_tendency: str             # "brief" | "moderate" | "verbose"
    formality: str                   # "casual" | "mixed" | "formal"
    tone: str                        # "warm" | "neutral" | "critical" | "enthusiastic"
    uses_pidgin: bool                # Nigerian Pidgin English detected
    signature_phrases: list[str]     # recurring phrases this user uses
    punctuation_style: str           # "minimal" | "expressive" (!!!, ???) | "standard"


@dataclass
class PreferenceProfile:
    top_categories: list[str]        # most reviewed categories
    loved_attributes: list[str]      # things they consistently praise
    disliked_attributes: list[str]   # things they consistently criticise
    deal_breakers: list[str]         # attributes that always cause low ratings
    must_haves: list[str]            # attributes that always cause high ratings
    cross_domain_interests: list[str] # interests inferred across categories


@dataclass
class BehaviouralProfile:
    review_frequency: str            # "occasional" | "regular" | "prolific"
    complain_ratio: float            # fraction of reviews that are negative (< 3 stars)
    detail_orientation: str          # "surface" | "detailed" (mentions specifics)
    social_signals: list[str]        # e.g. ["mentions friends", "mentions family visits"]
    cultural_context: list[str]      # Nigerian cultural cues detected


@dataclass
class UserProfile:
    user_id: str
    num_reviews: int
    rating: RatingProfile
    style: StyleProfile
    preferences: PreferenceProfile
    behaviour: BehaviouralProfile
    raw_summary: str                 # LLM's free-text synthesis
    confidence: str                  # "high" | "medium" | "low" (based on num_reviews)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """
        Returns a compact, human-readable string suitable for injecting
        into Task A and Task B prompts as user context.
        """
        p = self.preferences
        s = self.style
        r = self.rating
        b = self.behaviour

        lines = [
            f"USER PROFILE (id: {self.user_id}, based on {self.num_reviews} reviews, confidence: {self.confidence})",
            "",
            "RATING BEHAVIOUR",
            f"  Average rating: {r.mean:.1f}/5 | Tendency: {r.tendency} | Context sensitivity: {r.context_sensitivity}",
            "",
            "WRITING STYLE",
            f"  Length: {s.length_tendency} (~{s.avg_length_words} words) | Tone: {s.tone} | Formality: {s.formality}",
            f"  Uses Nigerian Pidgin: {'Yes' if s.uses_pidgin else 'No'}",
            f"  Signature phrases: {', '.join(s.signature_phrases[:5]) if s.signature_phrases else 'none detected'}",
            "",
            "PREFERENCES",
            f"  Top categories: {', '.join(p.top_categories[:4])}",
            f"  Loves: {', '.join(p.loved_attributes[:5])}",
            f"  Dislikes: {', '.join(p.disliked_attributes[:5])}",
            f"  Deal-breakers: {', '.join(p.deal_breakers[:3])}",
            f"  Must-haves: {', '.join(p.must_haves[:3])}",
            "",
            "BEHAVIOURAL SIGNALS",
            f"  Review frequency: {b.review_frequency} | Complaint ratio: {b.complain_ratio:.0%}",
            f"  Detail orientation: {b.detail_orientation}",
        ]
        if b.cultural_context:
            lines.append(f"  Cultural context: {', '.join(b.cultural_context)}")
        lines += ["", "SYNTHESIS", f"  {self.raw_summary}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def _compute_rating_stats(reviews: list[dict]) -> tuple[float, float, dict]:
    ratings = [float(r["rating"]) for r in reviews if "rating" in r]
    if not ratings:
        return 3.0, 0.0, {}
    mean = sum(ratings) / len(ratings)
    variance = sum((x - mean) ** 2 for x in ratings) / len(ratings)
    std = variance ** 0.5
    dist = {str(i): ratings.count(i) for i in range(1, 6)}
    return round(mean, 2), round(std, 2), dist


def _rating_tendency(mean: float, std: float, dist: dict) -> tuple[str, str]:
    tendency = (
        "generous" if mean >= 4.2 else
        "harsh" if mean <= 2.8 else
        "bimodal" if (dist.get("1", 0) + dist.get("2", 0)) > 0
                     and (dist.get("4", 0) + dist.get("5", 0)) > 0
                     and std > 1.4 else
        "balanced"
    )
    context_sensitivity = "high" if std > 1.2 else "low"
    return tendency, context_sensitivity


def build_user_profile(
    user_id: str,
    reviews: list[dict],
    client: Optional[anthropic.Anthropic] = None,
) -> UserProfile:
    """
    Main entry point. Takes a list of review dicts and returns a UserProfile.

    Each review dict should have:
      - "text": str          review body
      - "rating": float      star rating (1–5)
      - "category": str      business/product category (optional)
      - "item_name": str     name of reviewed item (optional)

    The LLM handles qualitative extraction; stats are computed deterministically.
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if not reviews:
        raise ValueError("Cannot build profile from empty review list.")

    # --- Deterministic stats ---
    mean, std, dist = _compute_rating_stats(reviews)
    tendency, ctx_sensitivity = _rating_tendency(mean, std, dist)
    confidence = (
        "high" if len(reviews) >= 20 else
        "medium" if len(reviews) >= 8 else
        "low"
    )

    # --- LLM qualitative extraction ---
    # Truncate to most recent 30 reviews to stay within context limits
    sample = reviews[-30:]
    review_text_block = "\n\n".join(
        f"[Review {i+1}] Rating: {r.get('rating', '?')}/5 | "
        f"Category: {r.get('category', 'unknown')} | "
        f"Item: {r.get('item_name', 'unknown')}\n"
        f"{r.get('text', '')}"
        for i, r in enumerate(sample)
    )

    system_prompt = """You are an expert behavioural analyst specialising in consumer psychology 
and review platform data. You extract structured user personas from review histories.

You must respond ONLY with a valid JSON object — no preamble, no markdown fences.
Be specific and evidence-based. Quote actual phrases from reviews where relevant.
Nigerian cultural context is important: detect Pidgin English, Nigerian idioms, 
local references (e.g. mentions of local foods, "customer is king" attitude, 
direct/indirect communication styles common in Nigerian consumer culture)."""

    extraction_prompt = f"""Analyse this user's review history and extract a detailed behavioural profile.

REVIEW HISTORY ({len(sample)} reviews, user_id: {user_id}):
{review_text_block}

Return a JSON object with EXACTLY this structure:
{{
  "style": {{
    "avg_length_words": <integer>,
    "length_tendency": "<brief|moderate|verbose>",
    "formality": "<casual|mixed|formal>",
    "tone": "<warm|neutral|critical|enthusiastic>",
    "uses_pidgin": <true|false>,
    "signature_phrases": ["<phrase1>", "<phrase2>", ...],
    "punctuation_style": "<minimal|expressive|standard>"
  }},
  "preferences": {{
    "top_categories": ["<cat1>", ...],
    "loved_attributes": ["<attr1>", ...],
    "disliked_attributes": ["<attr1>", ...],
    "deal_breakers": ["<attr1>", ...],
    "must_haves": ["<attr1>", ...]
  }},
  "behaviour": {{
    "review_frequency": "<occasional|regular|prolific>",
    "complain_ratio": <float 0.0-1.0>,
    "detail_orientation": "<surface|detailed>",
    "social_signals": ["<signal1>", ...],
    "cultural_context": ["<cue1>", ...]
  }},
  "cross_domain_interests": ["<interest1>", ...],
  "raw_summary": "<2-3 sentence synthesis of who this person is as a consumer>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": extraction_prompt}],
    )

    raw_json = response.content[0].text.strip()
    # Strip markdown fences if model wraps anyway
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    data = json.loads(raw_json)

    # --- Assemble the profile ---
    s = data["style"]
    p = data["preferences"]
    b = data["behaviour"]

    return UserProfile(
        user_id=user_id,
        num_reviews=len(reviews),
        rating=RatingProfile(
            mean=mean,
            std=std,
            distribution=dist,
            tendency=tendency,
            context_sensitivity=ctx_sensitivity,
        ),
        style=StyleProfile(
            avg_length_words=s["avg_length_words"],
            length_tendency=s["length_tendency"],
            formality=s["formality"],
            tone=s["tone"],
            uses_pidgin=s["uses_pidgin"],
            signature_phrases=s.get("signature_phrases", []),
            punctuation_style=s["punctuation_style"],
        ),
        preferences=PreferenceProfile(
            top_categories=p["top_categories"],
            loved_attributes=p["loved_attributes"],
            disliked_attributes=p["disliked_attributes"],
            deal_breakers=p["deal_breakers"],
            must_haves=p["must_haves"],
            cross_domain_interests=data.get("cross_domain_interests", []),
        ),
        behaviour=BehaviouralProfile(
            review_frequency=b["review_frequency"],
            complain_ratio=b["complain_ratio"],
            detail_orientation=b["detail_orientation"],
            social_signals=b.get("social_signals", []),
            cultural_context=b.get("cultural_context", []),
        ),
        raw_summary=data["raw_summary"],
        confidence=confidence,
    )
