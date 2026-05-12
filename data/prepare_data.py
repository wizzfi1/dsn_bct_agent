"""
Data Preparation Script
========================
ONE-TIME RUN — zero API calls, zero cost.

What this does:
  1. Loads Yelp, Amazon, and Goodreads raw datasets
  2. Finds users with rich review histories
  3. Creates small train/test splits saved to disk
  4. All future work reads from these small cached files

After running this, your datasets/ folder will have:
  datasets/
  └── prepared/
      ├── yelp_users.json          # top 200 users with full review history
      ├── yelp_train.json          # training reviews (80%)
      ├── yelp_test.json           # held-out reviews for evaluation (20%)
      ├── yelp_items.json          # business/item catalog for Task B
      ├── amazon_users.json        # top 100 amazon users
      ├── amazon_train.json
      ├── amazon_test.json
      ├── amazon_items.json
      └── stats.json               # dataset statistics summary

Usage:
  python data/prepare_data.py
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — update paths if needed
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "prepared"

YELP_REVIEW_PATH   = DATASETS_DIR / "Yelp-JSON" / "Yelp JSON" / "yelp_academic_dataset_review.json"
YELP_BUSINESS_PATH = DATASETS_DIR / "Yelp-JSON" / "Yelp JSON" / "yelp_academic_dataset_business.json"

AMAZON_DIR         = DATASETS_DIR / "amazon_food" / "full"

GOODREADS_REVIEWS  = DATASETS_DIR / "goodreads_reviews_dedup.json" / "goodreads_reviews_dedup.json"
GOODREADS_BOOKS    = DATASETS_DIR / "goodreads_books.json" / "goodreads_books.json"

# Tuning knobs
MIN_REVIEWS_PER_USER = 15       # minimum reviews to be a "rich" user
MAX_USERS            = 200      # how many users to keep for Yelp
MAX_USERS_AMAZON     = 100      # how many users to keep for Amazon
MAX_REVIEWS_SCAN     = 500_000  # stop scanning after this many raw reviews (saves time)
TEST_RATIO           = 0.2      # 20% held out for evaluation
RANDOM_SEED          = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {path.name} ({len(data)} items, {size_kb:.0f} KB)")


def load_jsonl(path: Path, max_rows: int = None) -> list[dict]:
    """Load a newline-delimited JSON file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def train_test_split(reviews: list[dict], test_ratio: float = 0.2, seed: int = 42):
    """
    Split reviews into train/test PER USER.
    Each user's most recent reviews go to test, older ones to train.
    This simulates the real evaluation scenario.
    """
    random.seed(seed)

    # Group by user
    by_user = defaultdict(list)
    for r in reviews:
        by_user[r["user_id"]].append(r)

    train, test = [], []
    for uid, user_reviews in by_user.items():
        # Sort by date if available, else shuffle
        try:
            user_reviews.sort(key=lambda x: x.get("date", ""), reverse=False)
        except Exception:
            pass

        n_test = max(1, int(len(user_reviews) * test_ratio))
        train.extend(user_reviews[:-n_test])
        test.extend(user_reviews[-n_test:])

    return train, test


# ---------------------------------------------------------------------------
# Yelp preparation
# ---------------------------------------------------------------------------

def prepare_yelp():
    print("\n" + "="*60)
    print("PREPARING YELP DATASET")
    print("="*60)

    if not YELP_REVIEW_PATH.exists():
        print(f"  SKIP — file not found: {YELP_REVIEW_PATH}")
        return

    # Step 1: Load business catalog
    print("\n[1/4] Loading business catalog...")
    business_map = {}
    for b in load_jsonl(YELP_BUSINESS_PATH):
        business_map[b["business_id"]] = {
            "item_id":    b["business_id"],
            "item_name":  b.get("name", ""),
            "category":   (b.get("categories") or "").split(",")[0].strip(),
            "city":       b.get("city", ""),
            "state":      b.get("state", ""),
            "stars":      b.get("stars", 0),
            "review_count": b.get("review_count", 0),
            "attributes": list((b.get("attributes") or {}).keys())[:5],
            "price_range": (b.get("attributes") or {}).get("RestaurantsPriceRange2", ""),
        }
    print(f"  Loaded {len(business_map):,} businesses")

    # Step 2: Scan reviews and build user index
    print(f"\n[2/4] Scanning reviews (up to {MAX_REVIEWS_SCAN:,})...")
    user_reviews = defaultdict(list)
    total = 0

    with open(YELP_REVIEW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if total >= MAX_REVIEWS_SCAN:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            biz = business_map.get(r.get("business_id", ""), {})
            review = {
                "user_id":   r.get("user_id", ""),
                "item_id":   r.get("business_id", ""),
                "item_name": biz.get("item_name", ""),
                "category":  biz.get("category", ""),
                "city":      biz.get("city", ""),
                "rating":    float(r.get("stars", 3)),
                "text":      r.get("text", ""),
                "date":      r.get("date", ""),
                "useful":    r.get("useful", 0),
                "source":    "yelp",
            }
            if review["user_id"] and review["text"]:
                user_reviews[review["user_id"]].append(review)
            total += 1

            if total % 100_000 == 0:
                print(f"  Scanned {total:,} reviews, {len(user_reviews):,} unique users...")

    print(f"  Total scanned: {total:,} reviews, {len(user_reviews):,} unique users")

    # Step 3: Select rich users
    print(f"\n[3/4] Selecting top {MAX_USERS} users with {MIN_REVIEWS_PER_USER}+ reviews...")
    qualified = [
        (uid, reviews)
        for uid, reviews in user_reviews.items()
        if len(reviews) >= MIN_REVIEWS_PER_USER
    ]
    qualified.sort(key=lambda x: len(x[1]), reverse=True)
    selected = qualified[:MAX_USERS]

    print(f"  Qualified users: {len(qualified):,}")
    print(f"  Selected: {len(selected)}")
    if selected:
        print(f"  Review counts: min={len(selected[-1][1])}, max={len(selected[0][1])}, "
              f"avg={sum(len(r) for _, r in selected)//len(selected)}")

    # Step 4: Build and save outputs
    print("\n[4/4] Saving prepared data...")

    all_reviews = [r for _, reviews in selected for r in reviews]
    train_reviews, test_reviews = train_test_split(all_reviews, TEST_RATIO, RANDOM_SEED)

    # User profiles index
    users_data = [
        {
            "user_id": uid,
            "num_reviews": len(reviews),
            "avg_rating": round(sum(r["rating"] for r in reviews) / len(reviews), 2),
            "categories": list({r["category"] for r in reviews if r["category"]})[:8],
        }
        for uid, reviews in selected
    ]

    # Item catalog (businesses seen by selected users)
    seen_item_ids = {r["item_id"] for r in all_reviews}
    items_data = [
        v for k, v in business_map.items()
        if k in seen_item_ids and v["item_name"]
    ]

    save_json(users_data,    OUTPUT_DIR / "yelp_users.json")
    save_json(train_reviews, OUTPUT_DIR / "yelp_train.json")
    save_json(test_reviews,  OUTPUT_DIR / "yelp_test.json")
    save_json(items_data,    OUTPUT_DIR / "yelp_items.json")

    print(f"\n  Train reviews: {len(train_reviews):,}")
    print(f"  Test reviews:  {len(test_reviews):,}")
    print(f"  Items catalog: {len(items_data):,}")

    return {
        "users": len(selected),
        "train": len(train_reviews),
        "test": len(test_reviews),
        "items": len(items_data),
    }


# ---------------------------------------------------------------------------
# Amazon preparation
# ---------------------------------------------------------------------------

def prepare_amazon():
    print("\n" + "="*60)
    print("PREPARING AMAZON DATASET")
    print("="*60)

    if not AMAZON_DIR.exists():
        print(f"  SKIP — directory not found: {AMAZON_DIR}")
        return

    # Load arrow files using datasets library
    try:
        from datasets import load_from_disk
    except ImportError:
        print("  SKIP — 'datasets' library not installed. Run: pip install datasets")
        return

    print("\n[1/4] Loading Amazon arrow files...")
    try:
        ds = load_from_disk(str(AMAZON_DIR.parent))
        # Handle DatasetDict (split into train/test/etc) vs Dataset
        if hasattr(ds, 'keys'):
            # DatasetDict — grab first available split
            split = list(ds.keys())[0]
            print(f"  DatasetDict detected, using split: '{split}'")
            ds = ds[split]
        print(f"  Dataset columns: {ds.column_names}")
        print(f"  Dataset size: {len(ds):,} rows")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    # Step 2: Build user index
    print(f"\n[2/4] Building user index (up to {MAX_REVIEWS_SCAN:,})...")
    user_reviews = defaultdict(list)

    for i in range(min(MAX_REVIEWS_SCAN, len(ds))):
        r = ds[i]
        # r is now guaranteed to be a dict
        if not isinstance(r, dict):
            continue
        uid = r.get("user_id", r.get("reviewerID", ""))
        text = r.get("text", r.get("reviewText", ""))
        if not uid or not text:
            continue

        review = {
            "user_id":   uid,
            "item_id":   r.get("parent_asin", r.get("asin", "")),
            "item_name": r.get("title", r.get("summary", "")),
            "category":  "Food and Grocery",
            "rating":    float(r.get("rating", r.get("overall", 3.0))),
            "text":      text,
            "date":      str(r.get("timestamp", r.get("reviewTime", ""))),
            "source":    "amazon",
        }
        user_reviews[uid].append(review)

        if (i+1) % 100_000 == 0:
            print(f"  Processed {i:,}...")

    print(f"  Unique users: {len(user_reviews):,}")

    # Step 3: Select rich users
    print(f"\n[3/4] Selecting top {MAX_USERS_AMAZON} users...")
    qualified = [
        (uid, reviews) for uid, reviews in user_reviews.items()
        if len(reviews) >= MIN_REVIEWS_PER_USER
    ]
    qualified.sort(key=lambda x: len(x[1]), reverse=True)
    selected = qualified[:MAX_USERS_AMAZON]
    print(f"  Qualified: {len(qualified):,}, Selected: {len(selected)}")

    # Step 4: Save
    print("\n[4/4] Saving prepared data...")
    all_reviews = [r for _, reviews in selected for r in reviews]
    train_reviews, test_reviews = train_test_split(all_reviews, TEST_RATIO, RANDOM_SEED)

    users_data = [
        {
            "user_id": uid,
            "num_reviews": len(reviews),
            "avg_rating": round(sum(r["rating"] for r in reviews) / len(reviews), 2),
            "categories": ["Food and Grocery"],
        }
        for uid, reviews in selected
    ]

    # Build item catalog
    items_seen = {}
    for r in all_reviews:
        if r["item_id"] not in items_seen:
            items_seen[r["item_id"]] = {
                "item_id":    r["item_id"],
                "item_name":  r["item_name"],
                "category":   r["category"],
                "avg_rating": r["rating"],
                "source":     "amazon",
            }

    save_json(users_data,         OUTPUT_DIR / "amazon_users.json")
    save_json(train_reviews,      OUTPUT_DIR / "amazon_train.json")
    save_json(test_reviews,       OUTPUT_DIR / "amazon_test.json")
    save_json(list(items_seen.values()), OUTPUT_DIR / "amazon_items.json")

    print(f"\n  Train: {len(train_reviews):,} | Test: {len(test_reviews):,}")

    return {
        "users": len(selected),
        "train": len(train_reviews),
        "test": len(test_reviews),
    }


# ---------------------------------------------------------------------------
# Goodreads preparation
# ---------------------------------------------------------------------------

def prepare_goodreads():
    print("\n" + "="*60)
    print("PREPARING GOODREADS DATASET")
    print("="*60)

    if not GOODREADS_REVIEWS.exists():
        print(f"  SKIP — file not found: {GOODREADS_REVIEWS}")
        return

    # Load book metadata
    book_map = {}
    if GOODREADS_BOOKS.exists():
        print("[1/3] Loading book metadata...")
        for b in load_jsonl(GOODREADS_BOOKS, max_rows=200_000):
            book_map[b.get("book_id", "")] = {
                "item_name": b.get("title", ""),
                "category":  b.get("popular_shelves", [{}])[0].get("name", "books")
                             if b.get("popular_shelves") else "books",
            }
        print(f"  Loaded {len(book_map):,} books")

    print(f"[2/3] Scanning reviews (up to {MAX_REVIEWS_SCAN:,})...")
    user_reviews = defaultdict(list)
    total = 0

    with open(GOODREADS_REVIEWS, "r", encoding="utf-8") as f:
        for line in f:
            if total >= MAX_REVIEWS_SCAN:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            rating = float(r.get("rating", 0))
            text = r.get("review_text", "").strip()
            uid = r.get("user_id", "")
            if rating == 0 or not text or not uid:
                continue

            book = book_map.get(r.get("book_id", ""), {})
            review = {
                "user_id":   uid,
                "item_id":   r.get("book_id", ""),
                "item_name": book.get("item_name", ""),
                "category":  book.get("category", "books"),
                "rating":    rating,
                "text":      text,
                "date":      r.get("date_added", ""),
                "source":    "goodreads",
            }
            user_reviews[uid].append(review)
            total += 1

    print(f"  Scanned {total:,} valid reviews, {len(user_reviews):,} users")

    print("[3/3] Selecting and saving...")
    qualified = [
        (uid, reviews) for uid, reviews in user_reviews.items()
        if len(reviews) >= MIN_REVIEWS_PER_USER
    ]
    qualified.sort(key=lambda x: len(x[1]), reverse=True)
    selected = qualified[:100]

    all_reviews = [r for _, reviews in selected for r in reviews]
    train_reviews, test_reviews = train_test_split(all_reviews, TEST_RATIO, RANDOM_SEED)

    save_json(train_reviews, OUTPUT_DIR / "goodreads_train.json")
    save_json(test_reviews,  OUTPUT_DIR / "goodreads_test.json")

    print(f"  Train: {len(train_reviews):,} | Test: {len(test_reviews):,}")

    return {"users": len(selected), "train": len(train_reviews), "test": len(test_reviews)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("DSN BCT — Data Preparation Script")
    print("Zero API calls. This is pure data processing.\n")

    ensure_output_dir()

    stats = {}

  #  yelp_stats = prepare_yelp()
  #  if yelp_stats:
  #      stats["yelp"] = yelp_stats

   # amazon_stats = prepare_amazon()
   # if amazon_stats:
   #     stats["amazon"] = amazon_stats

    goodreads_stats = prepare_goodreads()
    if goodreads_stats:
        stats["goodreads"] = goodreads_stats

    # Save summary stats
    save_json(stats, OUTPUT_DIR / "stats.json")

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nPrepared files saved to: {OUTPUT_DIR}")
    print("\nSummary:")
    for source, s in stats.items():
        print(f"  {source}: {s.get('users', 0)} users, "
              f"{s.get('train', 0):,} train, {s.get('test', 0):,} test reviews")

    print("\nNext step: run the profile engine on prepared data.")
    print("  python data/run_profiles.py")