"""
test_loss.py

Tests compute_loss with hand-constructed scores where we KNOW what the
loss should roughly look like, to verify the function behaves correctly
-- not just that it runs without crashing.

Run on Discovery:
    cd ~/fed_multimodal_rec/src/training
    python test_loss.py
"""

import torch
from loss import compute_loss


def test_confident_correct_prediction_gives_low_loss():
    """
    If the model is very confident the positive item (index 0) is correct,
    loss should be close to 0.
    """
    # Index 0 (positive) has a MUCH higher score than all others.
    scores = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1]])
    loss = compute_loss(scores)

    assert loss.item() < 0.01, f"Expected near-zero loss for confident correct prediction, got {loss.item()}"
    print(f"PASS: confident correct prediction gives low loss ({loss.item():.6f})")


def test_confident_wrong_prediction_gives_high_loss():
    """
    If the model is very confident a NEGATIVE item is correct (high score
    on some index other than 0), loss should be large.
    """
    # Index 1 has the highest score, NOT index 0 (the true positive).
    scores = torch.tensor([[0.1, 10.0, 0.1, 0.1, 0.1]])
    loss = compute_loss(scores)

    assert loss.item() > 5.0, f"Expected large loss for confident wrong prediction, got {loss.item()}"
    print(f"PASS: confident wrong prediction gives high loss ({loss.item():.4f})")


def test_uniform_scores_give_moderate_loss():
    """
    If all scores are equal (model has no idea / hasn't learned anything
    yet -- this is roughly what loss looks like at the VERY START of
    training, before any learning has happened), loss should equal
    -log(1/num_candidates), the mathematically expected value for a
    uniform random guess.
    """
    num_candidates = 5
    scores = torch.zeros((1, num_candidates))  # all scores identical (0.0)
    loss = compute_loss(scores)

    expected_loss = -torch.log(torch.tensor(1.0 / num_candidates))
    assert torch.allclose(loss, expected_loss, atol=1e-5), (
        f"Expected uniform-guess loss of {expected_loss.item():.4f}, got {loss.item():.4f}"
    )
    print(f"PASS: uniform/untrained-model scores give expected baseline loss "
          f"({loss.item():.4f} = -log(1/{num_candidates}))")
    print(f"      This is a useful sanity check during actual training: if loss")
    print(f"      starts near this value and decreases, training is working.")


def test_batch_of_multiple_examples():
    """
    Verifies the loss correctly averages across a batch of several
    examples with DIFFERENT confidence levels, rather than only being
    tested on a single-example batch.
    """
    scores = torch.tensor([
        [10.0, 0.1, 0.1],   # confident correct
        [0.1, 10.0, 0.1],   # confident WRONG
    ])
    loss = compute_loss(scores)

    # Loss should be the AVERAGE of a near-zero loss and a large loss --
    # so somewhere clearly above 0 but likely well below the second
    # example's individual loss alone, since averaging pulls it down.
    assert 1.0 < loss.item() < 6.0, f"Expected averaged loss in a moderate range, got {loss.item()}"
    print(f"PASS: batch of mixed confident-correct + confident-wrong examples "
          f"gives a moderate averaged loss ({loss.item():.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Loss Function Tests")
    print("=" * 60)

    test_confident_correct_prediction_gives_low_loss()
    test_confident_wrong_prediction_gives_high_loss()
    test_uniform_scores_give_moderate_loss()
    test_batch_of_multiple_examples()

    print()
    print("=" * 60)
    print("All loss function tests passed.")
    print("=" * 60)
