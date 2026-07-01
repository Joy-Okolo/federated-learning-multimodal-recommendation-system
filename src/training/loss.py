"""
training/loss.py

THE LOSS FUNCTION
==================
Implements sampled softmax cross-entropy: given a batch of scores
(one positive + N sampled negatives per user, with the positive always
at index 0), computes how far the model's predicted ranking is from the
ideal ranking where the positive item scores highest.

INTERVIEW SUMMARY:
  "Our loss is a sampled softmax cross-entropy. For each training example
   we have 100 candidate scores -- 1 positive at index 0 and 99 sampled
   negatives. We apply softmax across those 100 scores to get a probability
   distribution, then compute cross-entropy against the label 'index 0 is
   correct'. This is mathematically identical to standard multi-class
   classification cross-entropy, just with a different candidate set on
   every single training example, since the negatives are freshly sampled
   per example rather than being a fixed set of classes."
"""

import torch
import torch.nn as nn


def compute_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scores: shape (B, num_candidates). scores[:, 0] is ALWAYS the
            positive candidate's score, by the contract established in
            our dataset/collate pipeline (positive item placed at index 0).

    Returns:
        A single scalar tensor: the mean cross-entropy loss across the batch.
    """
    batch_size = scores.shape[0]

    # The "label" for cross-entropy is just the INDEX of the correct class.
    # Since we guarantee the positive is always at index 0, every single
    # example in the batch has the same label: 0.
    # torch.zeros(batch_size, dtype=torch.long) creates a tensor like
    # [0, 0, 0, ..., 0] -- one "correct index" per example in the batch.
    labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)

    # nn.CrossEntropyLoss expects RAW SCORES (often called "logits") as
    # input, and internally applies softmax + negative-log-likelihood
    # itself -- we do NOT manually apply softmax beforehand. This is a
    # common point of confusion: combining a separate softmax with
    # CrossEntropyLoss would silently apply softmax TWICE, producing
    # mathematically wrong (overly small/flat) gradients.
    #
    # Internally, for each row, CrossEntropyLoss computes:
    #   loss_i = -log( exp(scores[i, label_i]) / sum_j exp(scores[i, j]) )
    # which is EXACTLY the formula we derived by hand earlier:
    #   loss = -log(probability assigned to the positive item)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(scores, labels)

    return loss
