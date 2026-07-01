"""
inspect_data.py

A quick, human sanity-check of the real data we just loaded via the
adapter. We're looking for anything that would indicate the data is
malformed before we build a training loop on top of it -- e.g. titles
that look like garbage, image URLs that obviously aren't real URLs,
timestamps that don't make sense, or duplicate/missing values where
we don't expect them.

Run from anywhere (uses absolute-ish relative paths from src/):
    cd ~/fed_multimodal_rec/src
    python inspect_data.py
"""

import pandas as pd

pd.set_option("display.max_colwidth", 60)  # don't truncate text too aggressively


def inspect_items(items_df: pd.DataFrame):
    print("=" * 70)
    print("ITEMS TABLE")
    print("=" * 70)
    print(f"Shape: {items_df.shape}")
    print(f"Columns: {list(items_df.columns)}")
    print()

    print("--- First 5 rows (title, image_url, category) ---")
    print(items_df[["item_id", "title", "image_url", "category"]].head(5).to_string())
    print()

    print("--- Missing value check ---")
    print(items_df.isna().sum())
    print()

    print("--- Duplicate item_id check ---")
    n_duplicates = items_df["item_id"].duplicated().sum()
    print(f"Duplicate item_ids: {n_duplicates} (should be 0 -- each item should appear once)")
    print()

    print("--- Image URL sanity check ---")
    valid_url_prefix = items_df["image_url"].str.startswith("http").sum()
    print(f"URLs starting with 'http': {valid_url_prefix} / {len(items_df)}")
    print()

    print("--- Price sanity check ---")
    print(items_df["price"].describe())
    print()

    print("--- Top 10 categories ---")
    print(items_df["category"].value_counts().head(10))
    print()


def inspect_interactions(interactions_df: pd.DataFrame, items_df: pd.DataFrame):
    print("=" * 70)
    print("INTERACTIONS TABLE")
    print("=" * 70)
    print(f"Shape: {interactions_df.shape}")
    print(f"Columns: {list(interactions_df.columns)}")
    print()

    print("--- First 10 rows for ONE user (to verify chronological sort) ---")
    sample_user = interactions_df["user_id"].iloc[0]
    user_rows = interactions_df[interactions_df["user_id"] == sample_user]
    print(f"User: {sample_user} ({len(user_rows)} interactions)")
    print(user_rows[["timestamp", "item_id", "signal"]].to_string())
    print()

    # Verify timestamps are actually non-decreasing for this user --
    # if our earlier sort_values(["user_id", "timestamp"]) worked correctly,
    # this should never print anything.
    is_sorted = (user_rows["timestamp"].diff().dropna() >= 0).all()
    print(f"Timestamps non-decreasing for this user: {is_sorted} (should be True)")
    print()

    print("--- Signal (rating) distribution ---")
    print(interactions_df["signal"].value_counts().sort_index())
    print()

    print("--- Every interaction's item_id exists in items table? ---")
    valid_items = set(items_df["item_id"])
    orphaned = (~interactions_df["item_id"].isin(valid_items)).sum()
    print(f"Orphaned interactions (item_id not found in items table): {orphaned} (should be 0)")
    print()

    print("--- Re-checking min_interactions guarantee AFTER image filtering ---")
    counts = interactions_df.groupby("user_id").size()
    print(counts.describe())
    n_below_5 = (counts < 5).sum()
    print(f"Users with fewer than 5 interactions (after image-filtering): {n_below_5}")
    print("(We discussed this could happen -- image-filtering can drop a user")
    print(" below the original min_interactions threshold. This count tells")
    print(" us exactly how many users are affected.)")


if __name__ == "__main__":
    items_df = pd.read_parquet("../data/items.parquet")
    interactions_df = pd.read_parquet("../data/interactions.parquet")

    inspect_items(items_df)
    inspect_interactions(interactions_df, items_df)
