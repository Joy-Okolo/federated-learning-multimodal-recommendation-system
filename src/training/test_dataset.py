"""
test_dataset.py

Tests EcommerceSequenceDataset using small, fully controlled fake data so
we can hand-verify exactly what training/val/test examples get built.

Run on Discovery:
    cd ~/fed_multimodal_rec/src/training
    python test_dataset.py
"""

import pandas as pd
import torch

from dataset import EcommerceSequenceDataset


def make_fake_data():
    """
    One user with a clean, easy-to-verify-by-hand sequence of 6 items:
    A -> B -> C -> D -> E -> F (chronological order, timestamps 1..6).
    A second user with exactly 4 items (the minimum workable case) to
    test that boundary explicitly.
    """
    interactions_df = pd.DataFrame([
        {"user_id": "u1", "item_id": "A", "timestamp": 1, "signal": 5.0, "review_text": ""},
        {"user_id": "u1", "item_id": "B", "timestamp": 2, "signal": 5.0, "review_text": ""},
        {"user_id": "u1", "item_id": "C", "timestamp": 3, "signal": 5.0, "review_text": ""},
        {"user_id": "u1", "item_id": "D", "timestamp": 4, "signal": 5.0, "review_text": ""},
        {"user_id": "u1", "item_id": "E", "timestamp": 5, "signal": 5.0, "review_text": ""},
        {"user_id": "u1", "item_id": "F", "timestamp": 6, "signal": 5.0, "review_text": ""},
        # u2: exactly 4 items -- minimum workable case
        {"user_id": "u2", "item_id": "A", "timestamp": 1, "signal": 5.0, "review_text": ""},
        {"user_id": "u2", "item_id": "C", "timestamp": 2, "signal": 5.0, "review_text": ""},
        {"user_id": "u2", "item_id": "E", "timestamp": 3, "signal": 5.0, "review_text": ""},
        {"user_id": "u2", "item_id": "F", "timestamp": 4, "signal": 5.0, "review_text": ""},
        # u3: only 3 items -- should be EXCLUDED entirely (below minimum)
        {"user_id": "u3", "item_id": "A", "timestamp": 1, "signal": 5.0, "review_text": ""},
        {"user_id": "u3", "item_id": "B", "timestamp": 2, "signal": 5.0, "review_text": ""},
        {"user_id": "u3", "item_id": "C", "timestamp": 3, "signal": 5.0, "review_text": ""},
    ])

    items_df = pd.DataFrame([
        {"item_id": iid, "title": f"Item {iid}", "description": f"Description of {iid}",
         "image_url": f"http://fake.com/{iid}.jpg", "price": 9.99, "category": "Test"}
        for iid in ["A", "B", "C", "D", "E", "F"]
    ])

    return interactions_df, items_df


def test_train_split_no_leakage():
    interactions_df, items_df = make_fake_data()
    ds = EcommerceSequenceDataset(interactions_df, items_df, split="train", num_negatives=2, seed=0)

    u1_examples = [ex for ex in ds.examples if ex["user_id"] == "u1"]
    # Expected from [A,B,C,D,E,F]: usable_sequence = [A,B,C,D] (E,F excluded)
    # -> ([A]->B), ([A,B]->C), ([A,B,C]->D) = 3 examples
    assert len(u1_examples) == 3, f"Expected 3 train examples for u1, got {len(u1_examples)}"

    targets = [ex["target"] for ex in u1_examples]
    assert targets == ["B", "C", "D"], f"Expected targets [B,C,D], got {targets}"

    # CRITICAL: verify E and F (the val/test targets) NEVER appear as a
    # training target, and never appear inside any training history either.
    for ex in u1_examples:
        assert ex["target"] not in ("E", "F"), f"LEAKAGE: val/test item used as train target: {ex}"
        assert "E" not in ex["history"] and "F" not in ex["history"], (
            f"LEAKAGE: val/test item used in train history: {ex}"
        )

    print("PASS: train split for u1 produces exactly 3 examples (B, C, D),")
    print("      with E and F never appearing anywhere in training data")


def test_val_and_test_splits():
    interactions_df, items_df = make_fake_data()
    val_ds = EcommerceSequenceDataset(interactions_df, items_df, split="val", num_negatives=2, seed=0)
    test_ds = EcommerceSequenceDataset(interactions_df, items_df, split="test", num_negatives=2, seed=0)

    u1_val = [ex for ex in val_ds.examples if ex["user_id"] == "u1"][0]
    u1_test = [ex for ex in test_ds.examples if ex["user_id"] == "u1"][0]

    assert u1_val["target"] == "E", f"Expected val target E, got {u1_val['target']}"
    assert u1_val["history"] == ["A", "B", "C", "D"], f"Unexpected val history: {u1_val['history']}"

    assert u1_test["target"] == "F", f"Expected test target F, got {u1_test['target']}"
    assert u1_test["history"] == ["A", "B", "C", "D", "E"], f"Unexpected test history: {u1_test['history']}"

    print("PASS: val target=E (history=[A,B,C,D]), test target=F (history=[A,B,C,D,E])")


def test_minimum_length_boundary():
    interactions_df, items_df = make_fake_data()
    train_ds = EcommerceSequenceDataset(interactions_df, items_df, split="train", num_negatives=2, seed=0)

    # u2 has exactly 4 items [A,C,E,F] -- minimum workable case.
    # usable_sequence for train = [A,C] (E,F excluded) -> ONE example: [A]->C
    u2_train = [ex for ex in train_ds.examples if ex["user_id"] == "u2"]
    assert len(u2_train) == 1, f"Expected exactly 1 train example for u2, got {len(u2_train)}"
    assert u2_train[0]["target"] == "C"

    # u3 has only 3 items -- should be COMPLETELY EXCLUDED (below n<4 threshold)
    u3_train = [ex for ex in train_ds.examples if ex["user_id"] == "u3"]
    assert len(u3_train) == 0, f"Expected u3 (3 items) to be fully excluded, got {len(u3_train)} examples"

    print("PASS: u2 (exactly 4 items) produces 1 train example; "
          "u3 (3 items, below minimum) is fully excluded")


def test_negative_sampling_excludes_history_and_target():
    interactions_df, items_df = make_fake_data()
    ds = EcommerceSequenceDataset(interactions_df, items_df, split="train", num_negatives=2, seed=0)

    # Find the example where history=[A,B,C], target=D (u1's 3rd train example)
    example = [ex for ex in ds.examples if ex["user_id"] == "u1" and ex["target"] == "D"][0]
    exclude_set = set(example["history"]) | {example["target"]}
    # exclude_set = {A, B, C, D}

    # Sample negatives MANY times to check none of them ever violate the exclusion
    for _ in range(50):
        negatives = ds._sample_negatives(exclude_set)
        assert len(negatives) == 2, f"Expected 2 negatives, got {len(negatives)}"
        for neg in negatives:
            assert neg not in exclude_set, (
                f"Negative sampling violated exclusion: sampled {neg}, "
                f"which is in the user's history/target {exclude_set}"
            )

    print("PASS: negative sampling never includes items from the user's own history/target")


def test_getitem_shape_contract():
    """
    Verifies __getitem__ returns data matching EXACTLY what Recommender.forward()
    expects -- this is the contract between our data pipeline and our model.
    """
    interactions_df, items_df = make_fake_data()
    ds = EcommerceSequenceDataset(
        interactions_df, items_df, split="train", max_seq_len=10, num_negatives=3, seed=0
    )

    example = ds[0]  # __getitem__ via indexing

    assert len(example["padding_mask"]) == 10, "padding_mask should be max_seq_len long"
    assert example["padding_mask"].dtype == torch.bool

    n_candidates = 1 + 3  # positive + num_negatives
    assert len(example["candidate_image_urls"]) == n_candidates
    assert len(example["candidate_titles"]) == n_candidates
    assert len(example["candidate_descriptions"]) == n_candidates

    # history lists should be padded out to max_seq_len too (with placeholders)
    assert len(example["history_image_urls"]) == example["history_length"], (
        "history lists should contain only REAL items -- padding is handled "
        "via padding_mask + history_length, not by padding these lists "
        "(Recommender.encode_history only iterates up to history_length)"
    )

    print(f"PASS: __getitem__ output shapes match Recommender's expected contract")
    print(f"      (history_length={example['history_length']}, "
          f"num_candidates={n_candidates}, padding_mask_len={len(example['padding_mask'])})")


if __name__ == "__main__":
    print("=" * 60)
    print("EcommerceSequenceDataset Tests")
    print("=" * 60)

    test_train_split_no_leakage()
    print()
    test_val_and_test_splits()
    print()
    test_minimum_length_boundary()
    print()
    test_negative_sampling_excludes_history_and_target()
    print()
    test_getitem_shape_contract()

    print()
    print("=" * 60)
    print("All dataset tests passed.")
    print("=" * 60)
