"""
training/collate.py

A custom collate function for use with torch.utils.data.DataLoader.

WHY THIS EXISTS:
DataLoader's default batching logic tries to stack individual examples
into tensors automatically. That works fine for examples that are already
uniform tensors, but our dataset's __getitem__ returns a dict mixing real
tensors (padding_mask) with plain Python lists of strings (image URLs,
titles, descriptions) -- DataLoader has no built-in way to sensibly combine
lists of strings across a batch. A custom collate function lets us define
EXACTLY how a list of individual examples becomes one batched structure,
matching precisely what Recommender.forward() expects to receive.
"""

import torch


def collate_fn(batch: list) -> dict:
    """
    Args:
        batch: a list of B dicts, each one the output of
               EcommerceSequenceDataset.__getitem__ for one example.

    Returns:
        A single dict where each value represents the WHOLE batch, in
        exactly the shape Recommender.forward() expects:
          - history_image_urls / titles / descriptions: list of B lists
          - padding_mask: tensor of shape (B, max_seq_len)
          - history_lengths: list of B ints
          - candidate_image_urls / titles / descriptions: list of B lists
    """
    return {
        "user_ids": [ex["user_id"] for ex in batch],

        "history_image_urls": [ex["history_image_urls"] for ex in batch],
        "history_titles": [ex["history_titles"] for ex in batch],
        "history_descriptions": [ex["history_descriptions"] for ex in batch],

        # padding_mask: each example has shape (max_seq_len,) already.
        # torch.stack combines a LIST of same-shaped tensors into one
        # tensor with a new leading batch dimension: B tensors of shape
        # (max_seq_len,) -> one tensor of shape (B, max_seq_len).
        "padding_mask": torch.stack([ex["padding_mask"] for ex in batch]),

        "history_lengths": [ex["history_length"] for ex in batch],

        "candidate_image_urls": [ex["candidate_image_urls"] for ex in batch],
        "candidate_titles": [ex["candidate_titles"] for ex in batch],
        "candidate_descriptions": [ex["candidate_descriptions"] for ex in batch],

        # Kept for evaluation/debugging -- not consumed by Recommender itself.
        "candidate_item_ids": [ex["candidate_item_ids"] for ex in batch],
    }
