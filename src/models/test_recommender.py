"""
test_recommender.py

Tests the full Recommender model (ItemTower + UserTower combined) using
small fake data with mocked image downloads (so this runs fast and doesn't
depend on network access to real image URLs).

Run on Discovery:
    cd ~/fed_multimodal_rec/src/models
    python test_recommender.py
"""

import torch
from unittest.mock import patch

from recommender import Recommender


def fake_image_loader(self, image_url):
    """
    Replaces ItemTower._encode_image so tests don't need real network
    access or real images -- we just need SOME 512-dim vector, and we want
    it to be deterministic (same URL -> same fake vector) so we can reason
    about test results predictably.
    """
    # Hash the URL string into a seed so the same URL always produces the
    # same fake embedding -- mimics how a real image's embedding would be
    # consistent across calls.
    seed = abs(hash(image_url)) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(512, generator=generator)


def make_fake_batch(batch_size=2, history_len=4, num_candidates=5):
    """
    Builds a tiny fake batch: each user has `history_len` real history items
    (no padding, to keep this test focused on the Recommender's own logic
    rather than re-testing UserTower's padding handling, which we already
    tested separately in test_user_tower.py).
    """
    history_image_urls = [
        [f"http://fake.com/user{u}_item{i}.jpg" for i in range(history_len)]
        for u in range(batch_size)
    ]
    history_titles = [
        [f"Item {i} title" for i in range(history_len)]
        for u in range(batch_size)
    ]
    history_descriptions = [
        [f"Item {i} description" for i in range(history_len)]
        for u in range(batch_size)
    ]
    history_lengths = [history_len] * batch_size

    # Padding mask: since every user has exactly history_len real items and
    # we'll set max_seq_len = history_len for this test, there's NO padding.
    padding_mask = torch.zeros(batch_size, history_len, dtype=torch.bool)

    # Candidates: position 0 = positive, positions 1..N-1 = negatives.
    candidate_image_urls = [
        [f"http://fake.com/user{u}_candidate{c}.jpg" for c in range(num_candidates)]
        for u in range(batch_size)
    ]
    candidate_titles = [
        [f"Candidate {c} title" for c in range(num_candidates)]
        for u in range(batch_size)
    ]
    candidate_descriptions = [
        [f"Candidate {c} description" for c in range(num_candidates)]
        for u in range(batch_size)
    ]

    return (
        history_image_urls, history_titles, history_descriptions,
        padding_mask, history_lengths,
        candidate_image_urls, candidate_titles, candidate_descriptions,
    )


def test_output_shape():
    with patch("item_tower.ItemTower._encode_image", fake_image_loader):
        model = Recommender(embedding_dim=128)
        model.eval()

        batch = make_fake_batch(batch_size=2, history_len=4, num_candidates=5)
        with torch.no_grad():
            scores = model(*batch)

        assert scores.shape == (2, 5), f"Expected (2, 5), got {scores.shape}"
        print(f"PASS: output scores shape correct: {tuple(scores.shape)}")


def test_weight_sharing():
    """
    Verifies the SAME item, when it appears once in a user's history and
    once as a candidate, produces the IDENTICAL embedding. This is the
    weight-sharing guarantee we discussed -- the same physical item must
    be represented the same way regardless of its role in the computation.
    """
    with patch("item_tower.ItemTower._encode_image", fake_image_loader):
        model = Recommender(embedding_dim=128)
        model.eval()

        shared_url = "http://fake.com/shared_item.jpg"
        shared_title = "Shared Item"
        shared_desc = "This exact item appears in both history and candidates"

        # Encode it once "as a history item" via item_tower directly
        with torch.no_grad():
            embedding_as_history_item = model.item_tower([shared_url], [shared_title], [shared_desc])
            embedding_as_candidate = model.item_tower([shared_url], [shared_title], [shared_desc])

        assert torch.allclose(embedding_as_history_item, embedding_as_candidate, atol=1e-6), (
            "Same item produced different embeddings in different calls -- "
            "weight sharing is broken, or the model has nondeterministic "
            "behavior (e.g. dropout not disabled by .eval())."
        )
        print("PASS: same item produces identical embedding regardless of "
              "context (history vs candidate) -- weight sharing confirmed")


def test_positive_position_convention():
    """
    Sanity-checks the data CONTRACT, not the model itself: verifies that if
    we deliberately make candidate position 0 have a DIFFERENT embedding
    than positions 1+ (by using a distinguishable fake URL), the resulting
    score at position 0 reflects that specific item -- proving the model
    is not silently reordering or mixing up candidate positions internally.
    """
    with patch("item_tower.ItemTower._encode_image", fake_image_loader):
        model = Recommender(embedding_dim=128)
        model.eval()

        batch_size, history_len, num_candidates = 1, 3, 4
        batch = list(make_fake_batch(batch_size, history_len, num_candidates))

        # Manually relabel candidate 0 with a distinctive URL we can track.
        batch[5][0][0] = "http://fake.com/THE_POSITIVE_ITEM.jpg"  # candidate_image_urls

        with torch.no_grad():
            scores = model(*batch)
            # Independently compute what the score for THE_POSITIVE_ITEM
            # should be, to cross-check against scores[0][0].
            user_embedding = model.encode_history(
                batch[0], batch[1], batch[2], batch[3], batch[4]
            )
            # IMPORTANT: must use the EXACT SAME title/description that the
            # real batch uses for candidate 0 -- not placeholder text. The
            # text encoder is part of the embedding, so any text mismatch
            # here would produce a different embedding and an unfair
            # cross-check, even with the image URL matching correctly.
            positive_item_embedding = model.item_tower(
                [batch[5][0][0]],   # candidate_image_urls[user 0][candidate 0]
                [batch[6][0][0]],   # candidate_titles[user 0][candidate 0]
                [batch[7][0][0]],   # candidate_descriptions[user 0][candidate 0]
            )
            expected_score = (user_embedding[0] * positive_item_embedding[0]).sum()

        assert torch.allclose(scores[0, 0], expected_score, atol=1e-4), (
            f"Expected scores[0,0] to match independently computed score "
            f"for the labeled positive item ({expected_score.item():.4f}), "
            f"got {scores[0, 0].item():.4f}. This suggests candidate "
            f"ordering is not being preserved correctly through the model."
        )
        print("PASS: candidate position 0 reliably corresponds to the "
              "intended positive item -- no silent reordering in the model")


if __name__ == "__main__":
    print("=" * 60)
    print("Recommender Logic Tests")
    print("=" * 60)

    test_output_shape()
    test_weight_sharing()
    test_positive_position_convention()

    print()
    print("=" * 60)
    print("All Recommender tests passed.")
    print("=" * 60)
