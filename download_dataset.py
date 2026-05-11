from datasets import load_dataset

ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Grocery_and_Gourmet_Food",
    trust_remote_code=True
)

ds.save_to_disk("data/amazon_food")