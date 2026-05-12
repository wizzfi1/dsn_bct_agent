from pathlib import Path
import json

CACHE_DIR = Path('datasets/cache')
PREPARED_DIR = Path('datasets/prepared')

with open(PREPARED_DIR / 'yelp_users.json') as f:
    users = json.load(f)

users = sorted(users, key=lambda u: u['num_reviews'], reverse=True)[:6]
for u in users:
    uid = u['user_id']
    cached = (CACHE_DIR / 'yelp' / f'{uid}.json').exists()
    print(f"{uid[:20]} | reviews: {u['num_reviews']} | cached: {cached}")