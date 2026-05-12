import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.user_profile import build_user_profile

PREPARED_DIR = BASE_DIR / "datasets" / "prepared"
CACHE_DIR    = BASE_DIR / "datasets" / "cache"


def cache_path(user_id: str, source: str) -> Path:
    return CACHE_DIR / source / f"{user_id}.json"


def is_cached(user_id: str, source: str) -> bool:
    return cache_path(user_id, source).exists()


def save_profile(profile, source: str):
    path = cache_path(profile.user_id, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)


def load_cached_profile(user_id: str, source: str):
    path = cache_path(user_id, source)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_cached_profiles(source: str) -> list:
    source_dir = CACHE_DIR / source
    if not source_dir.exists():
        return []
    return [p.stem for p in source_dir.glob("*.json")]


def build_profiles_for_source(source: str, limit: int = 20, delay: float = 0.5):
    print(f"\n{'='*60}", flush=True)
    print(f"BUILDING PROFILES — {source.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    users_file = PREPARED_DIR / f"{source}_users.json"
    train_file = PREPARED_DIR / f"{source}_train.json"

    if not users_file.exists():
        print(f"  SKIP — {users_file} not found.", flush=True)
        return

    with open(users_file, "r", encoding="utf-8") as f:
        users = json.load(f)

    with open(train_file, "r", encoding="utf-8") as f:
        all_train = json.load(f)

    # Index train reviews by user
    train_by_user = defaultdict(list)
    for r in all_train:
        train_by_user[r["user_id"]].append(r)

    # Sort by review count, filter out users with no train reviews
    users = sorted(users, key=lambda u: u["num_reviews"], reverse=True)
    users = [u for u in users if len(train_by_user.get(u["user_id"], [])) >= 10]
    users = users[:limit]

    already_cached = [u for u in users if is_cached(u["user_id"], source)]
    to_build       = [u for u in users if not is_cached(u["user_id"], source)]

    print(f"\n  Total users selected: {len(users)}", flush=True)
    print(f"  Already cached:       {len(already_cached)} (free)", flush=True)
    print(f"  Need API calls:       {len(to_build)}", flush=True)
    print(f"  Estimated cost:       ~${len(to_build) * 0.003:.2f}", flush=True)

    if not to_build:
        print("\n  All profiles already cached! Nothing to do.", flush=True)
        return

    print(f"\n  Building {len(to_build)} profiles...\n", flush=True)

    success, failed = 0, 0

    for i, user in enumerate(to_build, 1):
        uid = user["user_id"]
        reviews = train_by_user.get(uid, [])

        print(f"  [{i}/{len(to_build)}] Building profile for {uid[:12]}... "
              f"({len(reviews)} reviews)", end=" ", flush=True)

        try:
            profile = build_user_profile(user_id=uid, reviews=reviews)
            save_profile(profile, source)
            print(f"✓ ({profile.rating.tendency}, {profile.style.tone}, "
                  f"pidgin={'yes' if profile.style.uses_pidgin else 'no'})", flush=True)
            success += 1
        except Exception as e:
            print(f"✗ ERROR: {e}", flush=True)
            failed += 1

        if i < len(to_build):
            time.sleep(delay)

    print(f"\n  Done. Success: {success}, Failed: {failed}", flush=True)
    print(f"  Cached profiles now: {len(list_cached_profiles(source))}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and cache user profiles")
    parser.add_argument("--source", choices=["yelp", "amazon", "goodreads", "all"],
                        default="yelp", help="Dataset source to process")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max number of users to process (default: 20)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sources = ["yelp", "amazon", "goodreads"] if args.source == "all" else [args.source]

    for source in sources:
        build_profiles_for_source(source, limit=args.limit, delay=args.delay)

    print("\n✓ Profile building complete.", flush=True)