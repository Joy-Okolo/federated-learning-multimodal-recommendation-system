"""
build_real_datasets.py

Quick sanity run: builds train/val/test EcommerceSequenceDataset splits
from our REAL loaded data (not fake test data), and reports their sizes.
This confirms the dataset logic works correctly at real scale before we
build the full training loop around it.

Run:
    cd ~/fed_multimodal_rec/src/training
    python build_real_datasets.py
"""

import pandas as pd
from dataset import EcommerceSequenceDataset


def main():
    print("Loading real data...")
    interactions_df = pd.read_parquet("../../data/interactions.parquet")
    items_df = pd.read_parquet("../../data/items.parquet")
    print(f"Loaded {len(interactions_df):,} interactions, {len(items_df):,} items")
    print()

    train_ds = EcommerceSequenceDataset(interactions_df, items_df, split="train",
                                          max_seq_len=50, num_negatives=99, seed=42)
    val_ds = EcommerceSequenceDataset(interactions_df, items_df, split="val",
                                        max_seq_len=50, num_negatives=99, seed=42)
    test_ds = EcommerceSequenceDataset(interactions_df, items_df, split="test",
                                         max_seq_len=50, num_negatives=99, seed=42)

    print()
    print("--- Pulling one real example from the train set, to eyeball it ---")
    example = train_ds[0]
    print(f"User: {example['user_id']}")
    print(f"History length: {example['history_length']}")
    print(f"History titles (first 3): {example['history_titles'][:3]}")
    print(f"Positive candidate title (index 0): {example['candidate_titles'][0]}")
    print(f"Number of candidates: {len(example['candidate_titles'])}")
    print(f"Padding mask shape: {example['padding_mask'].shape}, "
          f"sum (num real positions): {(~example['padding_mask']).sum().item()}")


if __name__ == "__main__":
    main()
