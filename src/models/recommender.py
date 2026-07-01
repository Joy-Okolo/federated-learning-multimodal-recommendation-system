"""
models/recommender.py

THE FULL RECOMMENDER MODEL
============================
Combines the ItemTower and UserTower into one model that, given a user's
history plus a set of candidate items (one positive + many negatives),
produces a similarity score for each candidate.

INTERVIEW SUMMARY:
  "The Recommender wraps both towers. For a batch of training examples, it
   encodes the user's history through the Item Tower (to get per-item
   embeddings) and User Tower (to compress those into one user embedding),
   separately encodes a set of candidate items (the positive next-item plus
   sampled negatives) through the same Item Tower, then scores every
   candidate via dot product against the user embedding. The model itself
   doesn't compute the loss -- it just returns scores. The training loop
   decides how to turn those scores into a loss, which keeps the model
   reusable for both training and pure inference."

IMPORTANT DESIGN NOTE -- weight sharing:
  The SAME ItemTower instance is used to encode BOTH the items in the
  user's history AND the candidate items being scored. This is intentional
  and important: an item must be represented identically regardless of
  whether it appears in someone's history or as a candidate being ranked.
  If we used two separate ItemTower instances, "Blue Shirt as history" and
  "Blue Shirt as candidate" could end up with different embeddings purely
  due to different random initialization -- which would be incoherent.
  Sharing one instance guarantees consistency.
"""

import torch
import torch.nn as nn

from item_tower import ItemTower
from user_tower import UserTower


class Recommender(nn.Module):
    """
    Full two-tower recommender: combines ItemTower + UserTower and produces
    similarity scores between a user and a set of candidate items.
    """

    def __init__(self, embedding_dim: int = 128):
        super(Recommender, self).__init__()

        # ONE shared ItemTower instance -- used for both history items and
        # candidate items, for the consistency reason explained above.
        self.item_tower = ItemTower(embedding_dim=embedding_dim)

        # ONE UserTower instance -- there's only ever one "kind" of user
        # encoding needed, so no sharing subtlety here, just a single tower.
        self.user_tower = UserTower(embedding_dim=embedding_dim)

        self.embedding_dim = embedding_dim

    def encode_history(
        self,
        history_image_urls: list,
        history_titles: list,
        history_descriptions: list,
        padding_mask: torch.Tensor,
        history_lengths: list,
    ) -> torch.Tensor:
        """
        Encodes a BATCH of users' histories into user embeddings.

        WHY THIS IS ITS OWN METHOD:
        This is genuinely the fiddliest part of the whole model, because the
        ItemTower expects a flat list of items (it doesn't know about
        "users" or "sequences" at all -- it just embeds items one at a
        time). The UserTower, on the other hand, expects a structured,
        padded (batch, seq_len, embedding_dim) tensor. We have to bridge
        between these two shapes ourselves here.

        Args:
            history_image_urls: list of B lists, each containing that
                user's real (non-padded) item image URLs, oldest first.
                Variable length per user.
            history_titles: same structure, item titles
            history_descriptions: same structure, item descriptions
            padding_mask: shape (B, max_seq_len), boolean -- True = padding.
                Precomputed by the caller (the data loader / training loop),
                since IT already knows each user's real history length.
            history_lengths: list of B ints -- real (non-padded) length
                of each user's history. Used to know where to place each
                user's real item embeddings within the padded tensor.

        Returns:
            Tensor of shape (B, embedding_dim): one user embedding per user.
        """
        batch_size = len(history_image_urls)
        max_seq_len = padding_mask.shape[1]

        # Start with an all-zero tensor -- padding positions will correctly
        # remain zero vectors, exactly as the UserTower expects.
        history_embeddings = torch.zeros(batch_size, max_seq_len, self.embedding_dim)

        # We flatten ALL real items across ALL users in the batch into one
        # single list, run them through ItemTower ONCE (much more efficient
        # than calling ItemTower once per user in a loop -- remember, batching
        # is what makes GPU computation fast), then scatter the results back
        # into the correct (user, position) slots afterward.
        flat_image_urls = []
        flat_titles = []
        flat_descriptions = []
        # Tracks which (user_idx, position_idx) each flattened item belongs to,
        # so we know where to put its embedding back after encoding.
        placement_indices = []

        for user_idx in range(batch_size):
            n_real = history_lengths[user_idx]
            for position_idx in range(n_real):
                flat_image_urls.append(history_image_urls[user_idx][position_idx])
                flat_titles.append(history_titles[user_idx][position_idx])
                flat_descriptions.append(history_descriptions[user_idx][position_idx])
                placement_indices.append((user_idx, position_idx))

        # Encode every real item across the entire batch in ONE ItemTower call.
        flat_item_embeddings = self.item_tower(flat_image_urls, flat_titles, flat_descriptions)
        # flat_item_embeddings shape: (total_real_items_across_batch, embedding_dim)

        # Scatter each embedding back into its correct (user, position) slot.
        for flat_idx, (user_idx, position_idx) in enumerate(placement_indices):
            history_embeddings[user_idx, position_idx, :] = flat_item_embeddings[flat_idx]

        # Now history_embeddings is a properly shaped, properly populated
        # (B, max_seq_len, embedding_dim) tensor, ready for the UserTower.
        user_embeddings = self.user_tower(history_embeddings, padding_mask)
        return user_embeddings

    def forward(
        self,
        history_image_urls: list,
        history_titles: list,
        history_descriptions: list,
        padding_mask: torch.Tensor,
        history_lengths: list,
        candidate_image_urls: list,
        candidate_titles: list,
        candidate_descriptions: list,
    ) -> torch.Tensor:
        """
        Full forward pass: encode histories into user embeddings, encode
        candidates into item embeddings, score every (user, candidate) pair.

        Args:
            (history_* and padding_mask / history_lengths: same as encode_history)
            candidate_image_urls: list of B lists, each containing
                (1 positive + N negative) candidate image URLs for that user
            candidate_titles, candidate_descriptions: same structure

        Returns:
            scores: shape (B, num_candidates_per_user)
                scores[i][0] is always the score for the POSITIVE candidate
                (by convention -- the caller is responsible for putting the
                 positive item first in each user's candidate list).
                scores[i][1:] are scores for the negative candidates.
        """
        # ── Step 1: encode user histories ───────────────────────────────────
        user_embeddings = self.encode_history(
            history_image_urls, history_titles, history_descriptions,
            padding_mask, history_lengths,
        )
        # user_embeddings shape: (B, embedding_dim)

        # ── Step 2: encode candidates ────────────────────────────────────────
        batch_size = len(candidate_image_urls)
        num_candidates = len(candidate_image_urls[0])  # assume same count per user
        # (1 positive + N negatives, same N for every user in the batch --
        #  the training loop / data loader is responsible for guaranteeing
        #  this consistent shape across the batch)

        # Flatten all candidates across the whole batch, same trick as before:
        # one big ItemTower call is far more efficient than B separate calls.
        flat_candidate_image_urls = [url for user_cands in candidate_image_urls for url in user_cands]
        flat_candidate_titles = [t for user_cands in candidate_titles for t in user_cands]
        flat_candidate_descriptions = [d for user_cands in candidate_descriptions for d in user_cands]

        flat_candidate_embeddings = self.item_tower(
            flat_candidate_image_urls, flat_candidate_titles, flat_candidate_descriptions
        )
        # flat_candidate_embeddings shape: (B * num_candidates, embedding_dim)

        # Reshape back into per-user groups of candidates.
        candidate_embeddings = flat_candidate_embeddings.view(
            batch_size, num_candidates, self.embedding_dim
        )
        # candidate_embeddings shape: (B, num_candidates, embedding_dim)

        # ── Step 3: score every (user, candidate) pair via dot product ──────
        # user_embeddings: (B, embedding_dim) → unsqueeze to (B, 1, embedding_dim)
        # candidate_embeddings: (B, num_candidates, embedding_dim)
        # Element-wise multiply, then sum over the embedding dimension --
        # this is exactly what a batched dot product looks like in PyTorch.
        scores = (user_embeddings.unsqueeze(1) * candidate_embeddings).sum(dim=-1)
        # scores shape: (B, num_candidates)

        return scores
