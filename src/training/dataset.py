"""
training/dataset.py

Converts raw interactions.parquet + items.parquet into PyTorch-ready
training examples, implementing the leave-one-out split and negative
sampling strategy we discussed conceptually.

INTERVIEW SUMMARY:
  "We use leave-one-out evaluation: for each user, the last interaction is
   held out for testing, the second-to-last for validation, and everything
   before that is used to build training examples via a sliding window
   over the user's chronological history. For each training example, we
   sample a fixed number of random negative items from the full catalog,
   excluding items the user has actually interacted with, to avoid
   accidentally treating a true positive as a negative."
"""

import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class EcommerceSequenceDataset(Dataset):
    """
    Builds leave-one-out training/validation/test examples from raw
    interaction + item data.

    WHY EXTEND torch.utils.data.Dataset?
    PyTorch's Dataset is an abstract base class with a simple contract:
    implement __len__ (how many examples exist) and __getitem__ (given an
    index, return that example). Once we do this, PyTorch's DataLoader can
    automatically handle batching, shuffling, and parallel data loading for
    us -- we don't have to write any of that machinery ourselves.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        split: str,                  # "train", "val", or "test"
        max_seq_len: int = 50,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.items_df = items_df.set_index("item_id")  # fast lookup by item_id
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.split = split

        # All item_ids in the catalog -- used as the pool to sample
        # negatives from.
        self.all_item_ids = items_df["item_id"].tolist()

        # A SEPARATE random generator, seeded deterministically, rather
        # than using Python's global `random` module directly. This means
        # re-running this script produces the EXACT SAME negative samples
        # every time -- important for reproducibility when comparing
        # experiments (e.g. centralized vs. federated training later),
        # since differences in results should come from the MODEL/METHOD,
        # not from random noise in which negatives happened to get sampled.
        self.rng = random.Random(seed)

        # ── Build per-user example lists ────────────────────────────────────
        # self.examples will hold one dict per training/val/test example,
        # built once here at construction time (not recomputed every epoch).
        self.examples = []

        grouped = interactions_df.groupby("user_id")
        for user_id, user_rows in grouped:
            # user_rows is already chronologically sorted (we verified this
            # in inspect_data.py), but we sort again defensively here --
            # cheap insurance against this Dataset ever being handed
            # unsorted data from a different source in the future.
            user_rows = user_rows.sort_values("timestamp")
            item_sequence = user_rows["item_id"].tolist()

            n = len(item_sequence)
            if n < 4:
                # Need at least 4 items to have >=1 train example, 1 val,
                # 1 test (matches our earlier discussion: 6 items -> 3
                # train + 1 val + 1 test; the absolute minimum workable
                # case is n=4 -> 1 train + 1 val + 1 test). We already
                # know from inspect_data.py that only 1 user in our real
                # dataset has fewer than 5 total interactions, and even
                # that user has >= 4, so in practice this filter rarely
                # excludes anyone -- but it's a necessary safety check.
                continue

            if split == "test":
                # ONE example per user: history = everything except the
                # last item, target = the last item.
                history = item_sequence[:-1]
                target = item_sequence[-1]
                self.examples.append({"user_id": user_id, "history": history, "target": target})

            elif split == "val":
                # ONE example per user: history = everything except the
                # last TWO items, target = the second-to-last item.
                history = item_sequence[:-2]
                target = item_sequence[-2]
                self.examples.append({"user_id": user_id, "history": history, "target": target})

            else:  # split == "train"
                # MULTIPLE examples per user via a sliding window, but
                # critically, NEVER touching the last two items (those are
                # reserved exclusively for val/test -- this is what
                # prevents test-set leakage into training).
                # Recall our earlier worked example: for [A,B,C,D,E,F],
                # training examples are history=[A]->B, [A,B]->C, [A,B,C]->D
                # -- i.e. every position EXCEPT the val and test targets.
                usable_sequence = item_sequence[:-2]  # exclude val+test targets entirely
                for i in range(1, len(usable_sequence)):
                    history = usable_sequence[:i]
                    target = usable_sequence[i]
                    self.examples.append({"user_id": user_id, "history": history, "target": target})

        print(f"[{split}] Built {len(self.examples):,} examples "
              f"from {interactions_df['user_id'].nunique():,} users")

    def __len__(self):
        return len(self.examples)

    def _sample_negatives(self, exclude_item_ids: set) -> list:
        """
        Randomly samples self.num_negatives item_ids from the full catalog,
        making sure none of them are items the user has actually
        interacted with (exclude_item_ids).

        WHY EXCLUDE THE USER'S OWN ITEMS?
        If we accidentally sampled an item the user genuinely liked as a
        "negative" example, we'd be teaching the model that something the
        user actually wants is something they DON'T want -- actively
        corrupting the training signal. This is a subtle correctness
        requirement that's easy to overlook.
        """
        negatives = []
        # We loop and re-sample on collision rather than using a single
        # bulk random.sample() call, because exclude_item_ids can vary in
        # size per user and we want to guarantee EXACTLY num_negatives
        # valid negatives every time, even in the (extremely unlikely,
        # given our catalog has 56,000+ items) case of repeated collisions.
        while len(negatives) < self.num_negatives:
            candidate = self.rng.choice(self.all_item_ids)
            if candidate not in exclude_item_ids and candidate not in negatives:
                negatives.append(candidate)
        return negatives

    def _build_item_payload(self, item_id: str) -> dict:
        """
        Looks up an item's image_url, title, and description from the
        items table, given its item_id. Centralizing this lookup in one
        method avoids repeating the same .loc[] call in multiple places
        below.
        """
        row = self.items_df.loc[item_id]
        return {
            "item_id": item_id,
            "image_url": row["image_url"],
            "title": row["title"],
            "description": row["description"],
        }

    def __getitem__(self, idx: int) -> dict:
        """
        Returns ONE complete training/val/test example, fully assembled:
        padded history (as item payloads + a padding mask), and a list of
        candidates with the positive item FIRST (index 0), followed by
        num_negatives sampled negatives -- matching the exact contract
        Recommender.forward() expects.
        """
        example = self.examples[idx]
        history_item_ids = example["history"]
        target_item_id = example["target"]

        # ── Truncate history to max_seq_len, keeping the MOST RECENT items ──
        # Recall our earlier discussion: if a history is longer than our
        # cap, we keep the most recent items (closer to the end of the
        # list), not the oldest ones, since recent interactions are more
        # predictive.
        if len(history_item_ids) > self.max_seq_len:
            history_item_ids = history_item_ids[-self.max_seq_len:]

        real_len = len(history_item_ids)
        pad_len = self.max_seq_len - real_len

        # Build the padding mask: False for real positions, True for padding.
        # (Matches the convention UserTower expects, which we tested explicitly.)
        padding_mask = torch.tensor(
            [False] * real_len + [True] * pad_len, dtype=torch.bool
        )

        # Build history item payloads. For padding positions, we still
        # need SOMETHING in the list (so all the lists stay the same
        # length, max_seq_len) -- we use placeholder empty values, which
        # is safe because UserTower/Recommender never actually look at
        # content for positions marked True in padding_mask; only the
        # POSITION matters for those slots, and Recommender's
        # encode_history method only flattens and encodes the REAL items
        # (using history_lengths to know where to stop), so these
        # placeholders are never actually passed through the Item Tower.
        history_image_urls = []
        history_titles = []
        history_descriptions = []
        for item_id in history_item_ids:
            payload = self._build_item_payload(item_id)
            history_image_urls.append(payload["image_url"])
            history_titles.append(payload["title"])
            history_descriptions.append(payload["description"])

        # ── Build candidates: positive FIRST, then sampled negatives ────────
        exclude_set = set(history_item_ids) | {target_item_id}
        negative_item_ids = self._sample_negatives(exclude_set)

        candidate_item_ids = [target_item_id] + negative_item_ids
        candidate_image_urls = []
        candidate_titles = []
        candidate_descriptions = []
        for item_id in candidate_item_ids:
            payload = self._build_item_payload(item_id)
            candidate_image_urls.append(payload["image_url"])
            candidate_titles.append(payload["title"])
            candidate_descriptions.append(payload["description"])

        return {
            "user_id": example["user_id"],
            "history_image_urls": history_image_urls,
            "history_titles": history_titles,
            "history_descriptions": history_descriptions,
            "padding_mask": padding_mask,
            "history_length": real_len,
            "candidate_image_urls": candidate_image_urls,
            "candidate_titles": candidate_titles,
            "candidate_descriptions": candidate_descriptions,
            "candidate_item_ids": candidate_item_ids,  # kept for debugging/eval
        }
