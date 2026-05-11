"""
Dataset Loaders
===============
Unified interface for loading Yelp, Amazon Reviews, and Goodreads datasets
into the standard review dict format consumed by the User Profile Engine.

Standard review dict format:
  {
    "user_id":    str,
    "item_id":    str,
    "item_name":  str,
    "category":   str,
    "rating":     float,   # 1.0 – 5.0
    "text":       str,
    "date":       str,     # ISO format where available
    "source":     str,     # "yelp" | "amazon" | "goodreads"
  }
"""

import json
import gzip
import csv
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Yelp
# ---------------------------------------------------------------------------

def load_yelp_reviews(
    review_path: str,
    business_path: str,
    max_reviews: int = 100_000,
) -> Iterator[dict]:
    """
    Loads from Yelp Academic Dataset JSON files.
    review_path: path to yelp_academic_dataset_review.json
    business_path: path to yelp_academic_dataset_business.json
    """
    # Build business lookup for category + name enrichment
    business_map: dict[str, dict] = {}
    with open(business_path, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            business_map[b["business_id"]] = {
                "name": b.get("name", ""),
                "category": (b.get("categories") or "").split(",")[0].strip(),
            }

    count = 0
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_reviews:
                break
            r = json.loads(line)
            biz = business_map.get(r.get("business_id", ""), {})
            yield {
                "user_id": r.get("user_id", ""),
                "item_id": r.get("business_id", ""),
                "item_name": biz.get("name", ""),
                "category": biz.get("category", ""),
                "rating": float(r.get("stars", 3)),
                "text": r.get("text", ""),
                "date": r.get("date", ""),
                "source": "yelp",
            }
            count += 1


def load_yelp_user_reviews(
    review_path: str,
    business_path: str,
    user_id: str,
) -> list[dict]:
    """Load all reviews for a single Yelp user."""
    return [
        r for r in load_yelp_reviews(review_path, business_path, max_reviews=5_000_000)
        if r["user_id"] == user_id
    ]


# ---------------------------------------------------------------------------
# Amazon Reviews
# ---------------------------------------------------------------------------

def load_amazon_reviews(
    json_gz_path: str,
    max_reviews: int = 100_000,
) -> Iterator[dict]:
    """
    Loads from Amazon Review Dataset (.json.gz format).
    Works with any category file (Electronics, Books, etc.)
    """
    open_fn = gzip.open if json_gz_path.endswith(".gz") else open
    count = 0
    with open_fn(json_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if count >= max_reviews:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Normalise rating: Amazon uses 1.0-5.0 floats
            rating = float(r.get("overall", r.get("rating", 3.0)))
            yield {
                "user_id": r.get("reviewerID", r.get("user_id", "")),
                "item_id": r.get("asin", r.get("item_id", "")),
                "item_name": r.get("summary", r.get("title", "")),
                "category": r.get("category", ""),
                "rating": rating,
                "text": r.get("reviewText", r.get("text", "")),
                "date": r.get("reviewTime", r.get("date", "")),
                "source": "amazon",
            }
            count += 1


def load_amazon_user_reviews(
    json_gz_path: str,
    user_id: str,
) -> list[dict]:
    """Load all reviews for a single Amazon user."""
    return [
        r for r in load_amazon_reviews(json_gz_path, max_reviews=5_000_000)
        if r["user_id"] == user_id
    ]


# ---------------------------------------------------------------------------
# Goodreads
# ---------------------------------------------------------------------------

def load_goodreads_reviews(
    review_path: str,
    books_path: str = None,
    max_reviews: int = 100_000,
) -> Iterator[dict]:
    """
    Loads from Goodreads dataset JSON files.
    Optionally enriches with book metadata if books_path provided.
    """
    book_map: dict[str, dict] = {}
    if books_path:
        with open(books_path, "r", encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)
                book_map[b.get("book_id", "")] = {
                    "name": b.get("title", ""),
                    "category": b.get("popular_shelves", [{}])[0].get("name", "books")
                    if b.get("popular_shelves") else "books",
                }

    count = 0
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_reviews:
                break
            r = json.loads(line)
            book = book_map.get(r.get("book_id", ""), {})
            # Goodreads ratings are 0-5; treat 0 as no rating
            rating = float(r.get("rating", 0))
            if rating == 0:
                continue
            yield {
                "user_id": r.get("user_id", ""),
                "item_id": r.get("book_id", ""),
                "item_name": book.get("name", r.get("book_id", "")),
                "category": book.get("category", "books"),
                "rating": rating,
                "text": r.get("review_text", ""),
                "date": r.get("date_added", ""),
                "source": "goodreads",
            }
            count += 1


# ---------------------------------------------------------------------------
# User index builder (for fast lookup)
# ---------------------------------------------------------------------------

def build_user_index(
    reviews: list[dict],
    min_reviews: int = 5,
) -> dict[str, list[dict]]:
    """
    Groups reviews by user_id, filtering out users with too few reviews
    to build a meaningful profile from.
    Returns: {user_id: [review, ...]}
    """
    index: dict[str, list[dict]] = {}
    for r in reviews:
        uid = r["user_id"]
        if uid:
            index.setdefault(uid, []).append(r)
    return {uid: revs for uid, revs in index.items() if len(revs) >= min_reviews}


def find_rich_users(
    user_index: dict[str, list[dict]],
    min_reviews: int = 15,
    top_n: int = 100,
) -> list[str]:
    """
    Returns user_ids of the most prolific reviewers — best candidates
    for training and evaluation where review history is crucial.
    """
    qualified = [
        (uid, len(revs))
        for uid, revs in user_index.items()
        if len(revs) >= min_reviews
    ]
    qualified.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in qualified[:top_n]]


# ---------------------------------------------------------------------------
# Sample data generator (for development/testing without real datasets)
# ---------------------------------------------------------------------------

SAMPLE_REVIEWS = [
    {
        "user_id": "user_ng_001",
        "item_id": "biz_001",
        "item_name": "Chicken Republic Lekki",
        "category": "Fast Food",
        "rating": 4.0,
        "text": "The chicken was fresh and the service was fast! I go there every weekend with my family. "
                "The jollof rice na the best for that area abeg. Customer service could be better sha.",
        "date": "2024-03-15",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_002",
        "item_name": "Mr Biggs Victoria Island",
        "category": "Fast Food",
        "rating": 2.0,
        "text": "E don fall. The quality is not what it used to be. Waited 30 minutes for my order "
                "and the meat pie was cold. They need to do better.",
        "date": "2024-02-20",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_003",
        "item_name": "Kilimanjaro Restaurant",
        "category": "Nigerian Cuisine",
        "rating": 5.0,
        "text": "This place is top notch! The suya is fire and the ambience is wonderful. "
                "Took my wife there for our anniversary. The pepper soup was on point — "
                "exactly the kind of authentic taste you cannot get everywhere. Highly recommend!",
        "date": "2024-01-10",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_004",
        "item_name": "TFC Abuja",
        "category": "Fast Food",
        "rating": 3.0,
        "text": "Average. The chicken was okay but nothing special. Portion size small for the price. "
                "The staff were friendly though which I appreciated.",
        "date": "2023-12-05",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_005",
        "item_name": "Bukka Hut",
        "category": "Nigerian Cuisine",
        "rating": 5.0,
        "text": "Nothing beats this place for authentic Nigerian food abeg. "
                "The buka vibes is always there. Egusi soup, pounded yam, the whole package. "
                "Very affordable too. I don bring all my friends here come. 10/10 no cap.",
        "date": "2023-11-22",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_006",
        "item_name": "Lagos Beach Resort",
        "category": "Hotels & Travel",
        "rating": 4.0,
        "text": "Nice experience overall. The beach view was amazing and the rooms were clean. "
                "Breakfast could be more variety. Good for family trips. Would go back.",
        "date": "2023-10-14",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_007",
        "item_name": "Everyday Supermarket",
        "category": "Shopping",
        "rating": 2.0,
        "text": "The prices have gone up too much. And the AC is always off. "
                "I no understand this kind management. They should listen to customer complaints.",
        "date": "2023-09-30",
        "source": "yelp",
    },
    {
        "user_id": "user_ng_001",
        "item_id": "biz_008",
        "item_name": "Dominos Pizza Ikeja",
        "category": "Pizza",
        "rating": 4.0,
        "text": "Consistent quality. The online ordering works well and delivery was on time. "
                "Pepperoni pizza was great. Price is a bit steep but you get what you pay for.",
        "date": "2023-08-18",
        "source": "yelp",
    },
]
