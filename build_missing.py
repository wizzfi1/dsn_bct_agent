import os
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from core.user_profile import build_user_profile

PREPARED_DIR = Path('datasets/prepared')
CACHE_DIR    = Path('datasets/cache/yelp')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(PREPARED_DIR / 'yelp_train.json') as f:
    all_train = json.load(f)

train_by_user = defaultdict(list)
for r in all_train:
    train_by_user[r['user_id']].append(r)

missing = ['bYENop4BuQepBjM1-BI3', '-G7Zkl1wIWBBmD0KRy_s']

for uid in missing:
    reviews = train_by_user.get(uid, [])
    print(f"Building profile for {uid[:20]} ({len(reviews)} reviews)...", end=" ", flush=True)
    try:
        profile = build_user_profile(user_id=uid, reviews=reviews)
        out_path = CACHE_DIR / f"{uid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"✓ ({profile.rating.tendency}, {profile.style.tone})")
    except Exception as e:
        print(f"✗ ERROR: {e}")

print("Done.")