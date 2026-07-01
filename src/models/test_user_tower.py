"""
test_user_tower.py

Tests UserTower with small, fake data so we can verify:
  1. Output shape is correct: (batch_size, embedding_dim)
  2. Output is L2-normalized (norm = 1.0)
  3. The padding-aware "last real position" extraction logic is correct
     (this is the trickiest part of the forward pass, so we test it
     explicitly with a hand-checkable example)
  4. Causal masking doesn't crash and produces different output if you
     change a "future" item -- a concrete way to prove the model truly
     cannot see ahead

Run this on Discovery (needs real torch):
    cd ~/fed_multimodal_rec/src/models
    python test_user_tower.py
"""

import torch
from user_tower import UserTower, MAX_SEQ_LEN, EMBEDDING_DIM


def make_fake_batch(batch_size=3, real_lengths=(3, 8, 1)):
    """
    Builds a small fake batch of user histories for testing.

    real_lengths: how many REAL (non-padding) items each user has.
    For this test we use 3 users with 3, 8, and 1 real items respectively
    -- deliberately varied lengths to stress-test the padding logic,
    including the edge case of a user with only 1 real interaction.
    """
    assert len(real_lengths) == batch_size

    # Random item embeddings for the "real" portion of each sequence.
    # In the real pipeline these would come from the Item Tower; here we
    # use random vectors since we're only testing UserTower's own logic.
    history_embeddings = torch.zeros(batch_size, MAX_SEQ_LEN, EMBEDDING_DIM)
    padding_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.bool)
    # Start by assuming everything is padding (True), then mark real
    # positions as False below.

    for user_idx, n_real in enumerate(real_lengths):
        # Fill the first n_real positions with random "real" embeddings
        history_embeddings[user_idx, :n_real, :] = torch.randn(n_real, EMBEDDING_DIM)
        # Mark those positions as NOT padding
        padding_mask[user_idx, :n_real] = False

    return history_embeddings, padding_mask


def test_output_shape():
    tower = UserTower()
    tower.eval()  # avoid BatchNorm-style batch-size issues during testing
                  # (UserTower doesn't use BatchNorm, but .eval() is good
                  # habit anyway to disable dropout for deterministic testing)

    history_embeddings, padding_mask = make_fake_batch(batch_size=3, real_lengths=(3, 8, 1))

    with torch.no_grad():
        user_embeddings = tower(history_embeddings, padding_mask)

    assert user_embeddings.shape == (3, EMBEDDING_DIM), (
        f"Expected shape (3, {EMBEDDING_DIM}), got {user_embeddings.shape}"
    )
    print(f"PASS: output shape correct: {tuple(user_embeddings.shape)}")


def test_l2_normalized():
    tower = UserTower()
    tower.eval()

    history_embeddings, padding_mask = make_fake_batch(batch_size=3, real_lengths=(3, 8, 1))

    with torch.no_grad():
        user_embeddings = tower(history_embeddings, padding_mask)

    norms = user_embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4), (
        f"Expected all norms ~1.0, got {norms}"
    )
    print(f"PASS: all output embeddings are L2-normalized (norms={norms.tolist()})")


def test_last_real_position_logic():
    """
    Directly tests the "find last real position per user" math, independent
    of the transformer, since this is the part most likely to have an
    off-by-one bug.
    """
    real_lengths_input = torch.tensor([3, 8, 1])
    # Simulate the padding_mask this would produce: True (=padding) for
    # everything beyond each user's real length.
    batch_size = 3
    padding_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.bool)
    for i, n_real in enumerate(real_lengths_input):
        padding_mask[i, :n_real] = False

    # Re-implement just the extraction logic from forward(), in isolation,
    # to verify it independently of the rest of the model.
    real_lengths = (~padding_mask).sum(dim=1)
    last_real_position = real_lengths - 1

    expected_last_positions = torch.tensor([2, 7, 0])
    # user 0: 3 real items → indices 0,1,2 → last real index = 2
    # user 1: 8 real items → indices 0..7 → last real index = 7
    # user 2: 1 real item  → index 0      → last real index = 0

    assert torch.equal(last_real_position, expected_last_positions), (
        f"Expected {expected_last_positions.tolist()}, got {last_real_position.tolist()}"
    )
    print(f"PASS: last real position extraction correct: {last_real_position.tolist()}")


def test_causal_masking_blocks_future():
    """
    Proves causal masking actually works: if we change an item that comes
    AFTER a user's "current" position, the output embedding at the earlier
    position should NOT change. We test this by comparing the FULL
    transformer output (before we slice out just the last real position)
    at an early position, under two versions of the input that only differ
    in a LATER position.
    """
    tower = UserTower()
    tower.eval()

    torch.manual_seed(42)  # reproducibility for this comparison

    batch_size = 1
    history_embeddings, padding_mask = make_fake_batch(batch_size=1, real_lengths=(10,))

    # Make a second version where we change ONLY position 9 (the very last
    # real item) to something completely different.
    history_embeddings_modified = history_embeddings.clone()
    history_embeddings_modified[0, 9, :] = torch.randn(EMBEDDING_DIM) * 100  # drastically different

    # Run the transformer internals manually (bypassing the final slicing)
    # so we can compare ALL position outputs, not just the last one.
    with torch.no_grad():
        seq_len = history_embeddings.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = tower.position_embedding(position_ids)

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)

        padding_mask_float = torch.zeros_like(padding_mask, dtype=torch.float32)
        padding_mask_float = padding_mask_float.masked_fill(padding_mask, float("-inf"))

        out_original = tower.transformer_encoder(
            history_embeddings + pos_embeds, mask=causal_mask, src_key_padding_mask=padding_mask_float
        )
        out_modified = tower.transformer_encoder(
            history_embeddings_modified + pos_embeds, mask=causal_mask, src_key_padding_mask=padding_mask_float
        )

    # Position 3's output should be IDENTICAL between both runs, because
    # position 3 is only allowed to attend to positions 0-3 -- it can never
    # "see" position 9, which is the only thing that changed.
    position_3_unchanged = torch.allclose(
        out_original[0, 3, :], out_modified[0, 3, :], atol=1e-5
    )

    # Position 9's output, by contrast, SHOULD differ, since we directly
    # changed position 9's own input.
    position_9_changed = not torch.allclose(
        out_original[0, 9, :], out_modified[0, 9, :], atol=1e-5
    )

    assert position_3_unchanged, (
        "Causal masking is BROKEN: position 3's output changed when we "
        "modified position 9, meaning the model is illegally looking ahead."
    )
    assert position_9_changed, (
        "Sanity check failed: position 9's own output should change when "
        "we modify position 9's own input."
    )

    print("PASS: causal masking confirmed -- position 3 is unaffected by changes")
    print("      to position 9 (the future), while position 9 itself does change.")


if __name__ == "__main__":
    print("=" * 60)
    print("UserTower Logic Tests")
    print("=" * 60)

    test_output_shape()
    test_l2_normalized()
    test_last_real_position_logic()
    test_causal_masking_blocks_future()

    print()
    print("=" * 60)
    print("All UserTower tests passed.")
    print("=" * 60)
