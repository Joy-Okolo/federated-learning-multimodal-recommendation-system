"""
adapters/amazon_reviews_adapter.py

Converts the public Amazon Reviews 2023 dataset (McAuley Lab) into this
project's generic e-commerce schema (see src/schema.py). This dataset is
used purely as a public, realistic benchmark to validate the recommender --
the project itself is platform-agnostic. Swap in a different adapter here
(e.g. a Shopify export, a custom store's CSV) and the model/training code
downstream does not change at all.

Usage:
    python amazon_reviews_adapter.py --min_interactions 5 --max_users 20000 \
        --out_interactions ../../data/interactions.parquet \
        --out_items ../../data/items.parquet
"""

import argparse
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

CATEGORY_CONFIG = "Clothing_Shoes_and_Jewelry"  # which Amazon category to pull from


def _stream_and_filter_reviews(min_interactions: int, max_users: int, scan_limit: int) -> pd.DataFrame:
    print(f"Loading review stream ({CATEGORY_CONFIG})...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{CATEGORY_CONFIG}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    print("Pass 1/2: counting interactions per user...")
    user_counts = defaultdict(int)
    n_seen = 0
    for row in ds:
        user_counts[row["user_id"]] += 1
        n_seen += 1
        if n_seen % 200_000 == 0:
            print(f"  scanned {n_seen:,} reviews...")
        if n_seen >= scan_limit:
            print(f"  reached scan_limit={scan_limit:,}, stopping scan")
            break

    qualifying_users = {u for u, c in user_counts.items() if c >= min_interactions}
    print(f"Users with >= {min_interactions} interactions: {len(qualifying_users):,}")

    if len(qualifying_users) > max_users:
        qualifying_users = set(sorted(qualifying_users)[:max_users])
    print(f"Keeping {len(qualifying_users):,} users for this subsample")

    print("Pass 2/2: collecting reviews for qualifying users...")
    ds2 = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{CATEGORY_CONFIG}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    rows = []
    n_seen = 0
    for row in ds2:
        n_seen += 1
        if row["user_id"] in qualifying_users:
            rows.append({
                "user_id": row["user_id"],
                "item_id": row["parent_asin"],          # generic name, platform-specific source
                "timestamp": row["timestamp"],
                "signal": float(row["rating"]),           # generic name: interaction strength
                "review_text": row.get("text", "") or "",
            })
        if n_seen >= scan_limit:
            break
        if n_seen % 200_000 == 0:
            print(f"  scanned {n_seen:,} / collected {len(rows):,} so far...")

    df = pd.DataFrame(rows)
    print(f"Final interactions dataframe: {df.shape}")
    return df


def _attach_item_metadata(interactions_df: pd.DataFrame) -> pd.DataFrame:
    needed_ids = set(interactions_df["item_id"].unique())
    print(f"Need metadata for {len(needed_ids):,} unique items")

    print("Loading metadata stream...")
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{CATEGORY_CONFIG}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    meta_rows = {}
    n_seen = 0
    for row in meta_ds:
        n_seen += 1
        item_id = row.get("parent_asin")
        if item_id in needed_ids and item_id not in meta_rows:
            images = row.get("images", {}) or {}
            large_imgs = images.get("large", []) or []
            image_url = next((u for u in large_imgs if u), None)

            meta_rows[item_id] = {
                "item_id": item_id,
                "title": row.get("title", "") or "",
                "description": " ".join(row.get("description", []) or []),
                "image_url": image_url,
                "price": row.get("price"),
                "category": row.get("main_category", "") or "",
            }
        if n_seen % 500_000 == 0:
            print(f"  scanned {n_seen:,} metadata rows, matched {len(meta_rows):,}...")
        if len(meta_rows) == len(needed_ids):
            print("  found metadata for all needed items, stopping early")
            break

    items_df = pd.DataFrame(list(meta_rows.values()))
    print(f"Items dataframe (pre-image-filter): {items_df.shape}")
    return items_df


def load_as_generic_schema(min_interactions: int, max_users: int, scan_limit: int = 2_000_000):
    """
    Returns (interactions_df, items_df) in the generic schema defined in
    src/schema.py. Items with no image are dropped from items_df, and any
    interactions referencing a dropped item are dropped from interactions_df,
    so the two dataframes stay consistent with each other.
    """
    interactions_df = _stream_and_filter_reviews(min_interactions, max_users, scan_limit)
    items_df = _attach_item_metadata(interactions_df)

    before = len(items_df)
    items_df = items_df[items_df["image_url"].notna()].reset_index(drop=True)
    print(f"Dropped {before - len(items_df):,} items with no image")

    valid_item_ids = set(items_df["item_id"])
    before = len(interactions_df)
    interactions_df = interactions_df[interactions_df["item_id"].isin(valid_item_ids)]
    print(f"Dropped {before - len(interactions_df):,} interactions referencing image-less items")

    interactions_df = interactions_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return interactions_df, items_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--max_users", type=int, default=20000)
    parser.add_argument("--scan_limit", type=int, default=2_000_000,
                         help="How many reviews to scan in the streaming pass (laptop-friendly cap)")
    parser.add_argument("--out_interactions", type=str, default="../../data/interactions.parquet")
    parser.add_argument("--out_items", type=str, default="../../data/items.parquet")
    args = parser.parse_args()

    interactions_df, items_df = load_as_generic_schema(
        args.min_interactions, args.max_users, args.scan_limit
    )

    interactions_df.to_parquet(args.out_interactions, index=False)
    items_df.to_parquet(args.out_items, index=False)
    print(f"Saved {len(interactions_df):,} interactions -> {args.out_interactions}")
    print(f"Saved {len(items_df):,} items -> {args.out_items}")

    print("\n--- Quick sanity stats ---")
    print(f"Unique users: {interactions_df['user_id'].nunique():,}")
    print(f"Unique items: {interactions_df['item_id'].nunique():,}")
    print(f"Interactions per user (describe):\n{interactions_df.groupby('user_id').size().describe()}")


if __name__ == "__main__":
    main()
