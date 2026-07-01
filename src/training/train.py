"""
training/train.py

THE ACTUAL TRAINING LOOP
==========================
Ties everything together: loads real data, builds train/val DataLoaders,
runs the Recommender forward pass, computes loss, backpropagates, and
updates weights via the optimizer -- repeated across many batches and
epochs.

INTERVIEW SUMMARY:
  "Our training loop uses Adam optimization over the sampled softmax
   cross-entropy loss. Each epoch, we iterate over all training examples
   in shuffled batches, compute scores via the Recommender, calculate
   loss, backpropagate, and step the optimizer. After each epoch we
   evaluate on the validation set (without computing gradients) to track
   whether the model is genuinely learning to generalize, not just
   memorizing the training set."

USAGE:
    python train.py --epochs 5 --batch_size 32 --lr 1e-4
"""

import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from recommender import Recommender
from dataset import EcommerceSequenceDataset
from collate import collate_fn
from loss import compute_loss


def run_one_epoch(model, dataloader, optimizer, device, is_training: bool):
    """
    Runs ONE full pass over the given dataloader.

    WHY ONE FUNCTION FOR BOTH TRAINING AND VALIDATION?
    Training and validation share almost all the same logic (forward pass,
    loss computation) -- the only differences are: (1) whether we call
    .backward() and step the optimizer, and (2) whether the model is in
    .train() or .eval() mode (which matters for BatchNorm/Dropout, as we
    discussed earlier). Writing one shared function with an `is_training`
    flag avoids duplicating the forward-pass logic in two separate places,
    which would risk them silently drifting apart over time as the code evolves.

    Args:
        model: the Recommender instance
        dataloader: yields collated batches (see collate.py)
        optimizer: only used if is_training=True
        device: "cuda" or "cpu" -- NOTE: currently our ItemTower downloads
            images and runs CLIP/SentenceTransformer purely on CPU
            internally (we never .to(device) those components in this
            first version) -- only matters once we look at speed; flagged
            here honestly rather than silently glossed over.
        is_training: if True, computes gradients and updates weights.
            If False (validation), wraps everything in torch.no_grad().

    Returns:
        The average loss across all batches in this epoch.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    # torch.no_grad() vs no context manager: during validation, we NEVER
    # want PyTorch tracking gradients (it costs extra memory and compute
    # for no benefit, since we never call .backward() during validation).
    # Using a conditional context manager here keeps the code path
    # genuinely shared rather than duplicated.
    context = torch.no_grad() if not is_training else torch.enable_grad()

    with context:
        for batch_idx, batch in enumerate(dataloader):
            scores = model(
                history_image_urls=batch["history_image_urls"],
                history_titles=batch["history_titles"],
                history_descriptions=batch["history_descriptions"],
                padding_mask=batch["padding_mask"].to(device),
                history_lengths=batch["history_lengths"],
                candidate_image_urls=batch["candidate_image_urls"],
                candidate_titles=batch["candidate_titles"],
                candidate_descriptions=batch["candidate_descriptions"],
            )

            loss = compute_loss(scores)

            if is_training:
                # Three-step PyTorch training ritual, always in this order:
                optimizer.zero_grad()   # 1. clear gradients from the PREVIOUS step
                                         #    (PyTorch ACCUMULATES gradients by default;
                                         #    without this, gradients from old batches
                                         #    would incorrectly add onto new ones)
                loss.backward()         # 2. compute gradients via backpropagation
                optimizer.step()        # 3. update weights using those gradients

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                phase = "train" if is_training else "val"
                print(f"  [{phase}] batch {batch_idx}/{len(dataloader)} "
                      f"loss={loss.item():.4f}")

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_negatives", type=int, default=99)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--interactions_path", type=str, default="../../data/interactions.parquet")
    parser.add_argument("--items_path", type=str, default="../../data/items.parquet")
    parser.add_argument("--checkpoint_path", type=str, default="../../outputs/recommender_checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    interactions_df = pd.read_parquet(args.interactions_path)
    items_df = pd.read_parquet(args.items_path)
    print(f"Loaded {len(interactions_df):,} interactions, {len(items_df):,} items")

    train_dataset = EcommerceSequenceDataset(
        interactions_df, items_df, split="train",
        max_seq_len=args.max_seq_len, num_negatives=args.num_negatives,
    )
    val_dataset = EcommerceSequenceDataset(
        interactions_df, items_df, split="val",
        max_seq_len=args.max_seq_len, num_negatives=args.num_negatives,
    )

    # shuffle=True for TRAINING ONLY: we want the model to see examples in
    # a different random order each epoch, which helps prevent it from
    # learning spurious patterns related to data ORDER rather than actual
    # content. We do NOT shuffle validation -- order doesn't matter there
    # since we're not learning anything, just measuring performance, and
    # keeping it unshuffled makes debugging/comparing runs easier.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print("Initializing model...")
    model = Recommender(embedding_dim=128)
    # NOTE: model.to(device) deliberately NOT called yet -- see the honest
    # limitation noted in run_one_epoch's docstring. This first training
    # run validates CORRECTNESS on CPU; GPU optimization is a follow-up step.

    # Adam is the standard, default-choice optimizer for most deep learning
    # tasks. We only pass model.parameters() -- but recall, CLIP and the
    # Sentence Transformer have requires_grad=False, so even though they
    # technically show up when iterating model.parameters(), Adam will
    # simply never update them (no gradient ever flows there). Only the
    # fusion MLP and the UserTower's transformer/positional embeddings
    # actually get updated.
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")

        start_time = time.time()
        train_loss = run_one_epoch(model, train_loader, optimizer, device, is_training=True)
        train_time = time.time() - start_time

        start_time = time.time()
        val_loss = run_one_epoch(model, val_loader, optimizer, device, is_training=False)
        val_time = time.time() - start_time

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train loss: {train_loss:.4f} ({train_time:.1f}s)")
        print(f"  Val loss:   {val_loss:.4f} ({val_time:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"  New best val loss -- saved checkpoint to {args.checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
