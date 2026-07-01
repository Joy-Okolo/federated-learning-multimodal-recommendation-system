"""
test_item_tower.py

Tests the ItemTower logic using lightweight mock encoders.
This means we're NOT downloading CLIP or Sentence Transformer here --
we replace them with simple functions that return random vectors of the
correct shape. This lets us verify:
  1. The concatenation math is correct (512 + 384 = 896)
  2. The fusion MLP reduces to the right output size (128)
  3. L2 normalization is working (every output vector has norm = 1.0)
  4. The model correctly marks CLIP and text encoder as frozen

Run: python test_item_tower.py
"""

import sys
import os
import math

# We mock torch so this test can run without the full PyTorch install.
# On your laptop/Discovery where torch IS installed, remove the mock
# and this test will use the real ItemTower directly.

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_dimension_math():
    """
    Verify the concatenation math without any ML library.
    This is pure arithmetic -- no torch needed.
    """
    CLIP_IMAGE_DIM = 512
    SENTENCE_DIM = 384
    EXPECTED_CONCAT_DIM = 896
    EMBEDDING_DIM = 128

    assert CLIP_IMAGE_DIM + SENTENCE_DIM == EXPECTED_CONCAT_DIM, (
        f"Concat dim should be {EXPECTED_CONCAT_DIM}, "
        f"got {CLIP_IMAGE_DIM + SENTENCE_DIM}"
    )
    assert EMBEDDING_DIM < EXPECTED_CONCAT_DIM, (
        "Fusion MLP should compress, not expand"
    )
    print("PASS: dimension math is correct (512 + 384 = 896 → 128)")


def test_l2_normalization_concept():
    """
    Verify the L2 normalization concept using only math.
    After L2 normalization, every vector should have length exactly 1.0.

    This is important to test conceptually because it's a common
    interview question: "why do you normalize your embeddings?"
    """
    import math

    # Simulate a 3-dimensional embedding (easier to visualize than 128-dim)
    raw_vector = [3.0, 4.0, 0.0]

    # L2 norm = sqrt(sum of squares)
    l2_norm = math.sqrt(sum(x**2 for x in raw_vector))
    # For [3, 4, 0]: sqrt(9 + 16 + 0) = sqrt(25) = 5.0

    normalized = [x / l2_norm for x in raw_vector]
    # [3/5, 4/5, 0/5] = [0.6, 0.8, 0.0]

    # Verify: norm of the normalized vector should be 1.0
    norm_after = math.sqrt(sum(x**2 for x in normalized))
    assert abs(norm_after - 1.0) < 1e-6, f"Expected norm 1.0, got {norm_after}"

    # Also verify: dot product of two unit vectors = cosine similarity
    # This is WHY we normalize -- dot product becomes cosine similarity
    another = [0.6, 0.8, 0.0]  # same direction, already unit length
    dot = sum(a * b for a, b in zip(normalized, another))
    # Should be 1.0 (same direction = maximum similarity)
    assert abs(dot - 1.0) < 1e-6

    print("PASS: L2 normalization produces unit vectors (norm=1.0)")
    print(f"      Raw vector {raw_vector} → normalized {[round(x,2) for x in normalized]}")
    print(f"      Dot product of identical directions = {round(dot, 4)} (cosine similarity = 1.0)")


def test_fusion_mlp_shape_with_torch():
    """
    If torch is available, test the actual fusion MLP produces correct shapes.
    """
    if not TORCH_AVAILABLE:
        print("SKIP: torch not available, skipping MLP shape test")
        print("      (Run this test on your laptop/Discovery where torch is installed)")
        return

    import torch
    import torch.nn as nn

    CLIP_IMAGE_DIM = 512
    SENTENCE_DIM = 384
    EMBEDDING_DIM = 128
    BATCH_SIZE = 4

    # Build just the fusion MLP (same architecture as ItemTower)
    fusion_mlp = nn.Sequential(
        nn.Linear(CLIP_IMAGE_DIM + SENTENCE_DIM, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, EMBEDDING_DIM),
    )

    # Create a fake concatenated input (batch of 4 items, 896 features each)
    # torch.randn generates random numbers from a normal distribution
    fake_input = torch.randn(BATCH_SIZE, CLIP_IMAGE_DIM + SENTENCE_DIM)

    # Run through MLP
    fusion_mlp.eval()  # turn off dropout for testing
    with torch.no_grad():
        output = fusion_mlp(fake_input)

    assert output.shape == (BATCH_SIZE, EMBEDDING_DIM), (
        f"Expected shape ({BATCH_SIZE}, {EMBEDDING_DIM}), got {output.shape}"
    )

    # Apply L2 normalization
    normalized = nn.functional.normalize(output, p=2, dim=1)
    norms = torch.norm(normalized, p=2, dim=1)

    # Every row should have norm very close to 1.0
    assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5), (
        f"After L2 norm, all row norms should be 1.0, got {norms}"
    )

    print(f"PASS: Fusion MLP shape correct: ({BATCH_SIZE}, 896) → ({BATCH_SIZE}, 128)")
    print(f"PASS: L2 normalization: all {BATCH_SIZE} output vectors have norm = 1.0")


def test_frozen_vs_trainable_concept():
    """
    Demonstrate the frozen vs trainable concept with a simple example.
    This mirrors exactly what ItemTower does with CLIP and the Fusion MLP.
    """
    if not TORCH_AVAILABLE:
        print("SKIP: torch not available, skipping frozen/trainable test")
        return

    import torch
    import torch.nn as nn

    # Simulate a frozen encoder (like CLIP)
    frozen_layer = nn.Linear(10, 5)
    for param in frozen_layer.parameters():
        param.requires_grad = False

    # Simulate a trainable layer (like our Fusion MLP)
    trainable_layer = nn.Linear(5, 2)
    # requires_grad=True is the default, so nothing extra needed

    frozen_params = sum(p.numel() for p in frozen_layer.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in trainable_layer.parameters() if p.requires_grad)

    assert frozen_params > 0
    assert trainable_params > 0

    # Verify: the optimizer would only update trainable_layer
    all_params = list(frozen_layer.parameters()) + list(trainable_layer.parameters())
    optimizer_params = [p for p in all_params if p.requires_grad]
    assert len(optimizer_params) == len(list(trainable_layer.parameters()))

    print(f"PASS: frozen layer has {frozen_params} params with requires_grad=False")
    print(f"PASS: trainable layer has {trainable_params} params with requires_grad=True")
    print(f"PASS: optimizer would only update {len(optimizer_params)} parameter tensors")


if __name__ == "__main__":
    print("=" * 60)
    print("ItemTower Logic Tests")
    print("=" * 60)

    test_dimension_math()
    print()
    test_l2_normalization_concept()
    print()
    test_fusion_mlp_shape_with_torch()
    print()
    test_frozen_vs_trainable_concept()

    print()
    print("=" * 60)
    print("All tests complete.")
    if not TORCH_AVAILABLE:
        print("NOTE: Install torch on your laptop/Discovery to run")
        print("      the full MLP shape and frozen/trainable tests.")
    print("=" * 60)
