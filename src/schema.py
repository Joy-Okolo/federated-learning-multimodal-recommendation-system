"""
schema.py -- the generic, platform-agnostic data contract for this project.

DESIGN GOAL:
This project is a multimodal recommender for e-commerce in general, not for
any single platform. To keep it that way, every data source (Amazon Reviews,
a Shopify-style export, a custom CSV from any online store, etc.) must be
converted into these two generic shapes before touching the model or the
federated simulation. Nothing downstream of this file should ever import or
reference a platform-specific dataset name.

If you add a new data source later, write a new adapter in src/adapters/
that outputs these exact columns -- everything else (model, training,
federated client partitioning) just works unmodified.
"""

import pandas as pd

# Required columns for the interactions table.
# One row = one user interacting with one item at some point in time.
INTERACTION_COLUMNS = [
    "user_id",       # str: anonymized/platform-specific user identifier
    "item_id",       # str: unique item identifier (platform-agnostic key)
    "timestamp",     # int: unix timestamp (or any consistently sortable int)
    "signal",        # float: interaction strength (rating, click=1, purchase=1, etc.)
    "review_text",   # str, optional: free text associated with the interaction (may be "")
]

# Required columns for the item/catalog table.
# One row = one product in the catalog.
ITEM_COLUMNS = [
    "item_id",       # str: must match interactions.item_id
    "title",         # str: product title/name
    "description",   # str: product description (may be "")
    "image_url",      # str: URL or local path to a representative product image
    "price",         # float or None
    "category",      # str, optional: product category/taxonomy label
]


def validate_interactions(df: pd.DataFrame) -> None:
    missing = set(INTERACTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Interactions dataframe missing required columns: {missing}")


def validate_items(df: pd.DataFrame) -> None:
    missing = set(ITEM_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Items dataframe missing required columns: {missing}")
    if df["image_url"].isna().any():
        raise ValueError(
            "Items dataframe contains rows with missing image_url. "
            "Drop or impute these before using the multimodal model."
        )
