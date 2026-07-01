"""
models/user_tower.py

THE USER TOWER
===============
Takes a user's interaction history (a sequence of item embeddings, oldest
to most recent) and produces a single 128-dimensional user embedding.

INTERVIEW SUMMARY:
  "The user tower is a lightweight transformer encoder. Each item embedding
   in the user's history gets a positional embedding added to it so the
   model knows the order of interactions. We apply causal masking so that
   when computing the representation after item i, the model cannot see
   items i+1 onward -- this mirrors the real deployment scenario where we
   only ever know what happened up to now, never the future. We also use
   a padding mask because users have variable-length histories, and we pad
   shorter sequences to a fixed length of 50 so they can be batched together
   efficiently; the padding mask tells the model to ignore those filler
   positions entirely."
"""

import torch
import torch.nn as nn


# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 128       # must match the Item Tower's output dimension
MAX_SEQ_LEN = 50          # cap on history length (most recent 50 interactions kept)
NUM_ATTENTION_HEADS = 4   # see explanation below
NUM_TRANSFORMER_LAYERS = 1  # we keep this shallow -- explained in our discussion
                             # of why a single-layer transformer suits our dataset size
DROPOUT = 0.2


# ── User Tower Class ────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """
    Transformer-based sequence encoder: history of item embeddings → user embedding.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        max_seq_len: int = MAX_SEQ_LEN,
        num_heads: int = NUM_ATTENTION_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        dropout: float = DROPOUT,
    ):
        super(UserTower, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # ── Positional Embeddings ──────────────────────────────────────────
        # nn.Embedding is normally used in NLP to map word indices (integers)
        # to dense vectors. We're repurposing it here for POSITION indices
        # instead of word indices: position 0, 1, 2, ..., max_seq_len-1 each
        # get their own learnable 128-dim vector.
        #
        # WHY LEARNABLE, rather than a fixed mathematical formula (like the
        # sine/cosine positional encoding from the original "Attention is
        # All You Need" paper)?
        # The original transformer paper used a fixed sine/cosine formula
        # because it was designed for arbitrarily long text sequences and
        # needed to generalize to lengths never seen during training.
        # Our setting is different: we KNOW our max sequence length is fixed
        # at 50 (we capped it ourselves). With a fixed, known max length,
        # learnable positional embeddings are simpler to implement and tend
        # to perform at least as well in practice, since the model can learn
        # whatever positional pattern is actually useful for OUR specific
        # task rather than being constrained to a predetermined mathematical
        # shape.
        #
        # nn.Embedding(num_embeddings, embedding_dim):
        #   num_embeddings = max_seq_len → one embedding row per position
        #   embedding_dim = 128 → same size as item embeddings, so we can
        #                          add them together element-wise
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
        )

        # ── Transformer Encoder ─────────────────────────────────────────────
        # PyTorch gives us a built-in, well-tested transformer implementation
        # rather than writing attention math by hand. We build it in two steps:
        #
        # Step A: define ONE encoder layer's architecture (TransformerEncoderLayer)
        # Step B: stack `num_layers` copies of it (TransformerEncoder)
        #
        # TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first):
        #   d_model        = embedding_dim (128) -- the size of vectors flowing
        #                     through the whole layer; must match our item
        #                     embedding size so positions, items, and the
        #                     attention math all line up dimensionally
        #   nhead          = num_heads (4) -- the multi-head attention concept
        #                     we just discussed: 4 parallel "lenses" of 32
        #                     dims each (128 / 4 = 32)
        #   dim_feedforward = 256 -- inside each transformer layer, after the
        #                     attention step, there's a small feedforward
        #                     network (like a mini fusion-MLP) that further
        #                     transforms each position's representation.
        #                     256 = 2x the embedding_dim, a common convention
        #                     that gives the feedforward sublayer enough
        #                     capacity without being needlessly large
        #   dropout        = 0.2 -- same overfitting-prevention idea as in
        #                     the Item Tower's fusion MLP
        #   batch_first    = True -- IMPORTANT: this tells PyTorch our input
        #                     tensors will have shape (batch, sequence, features)
        #                     rather than (sequence, batch, features). Older
        #                     PyTorch defaults assumed sequence-first, which
        #                     is a very common source of silent shape bugs.
        #                     We explicitly set batch_first=True so our
        #                     tensors stay in the more intuitive
        #                     (batch, seq_len, embedding_dim) shape throughout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
        )

        # Stack `num_layers` copies of that layer. We set num_layers=1 (see
        # NUM_TRANSFORMER_LAYERS constant), so in our current configuration
        # this is really just one encoder layer -- but writing it this way
        # means increasing depth later (if we get more data, e.g. once we
        # scale up on Discovery) is a one-line config change, not a
        # structural rewrite.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: shape (B, max_seq_len, embedding_dim)
                B users, each with a padded sequence of up to max_seq_len
                item embeddings (oldest item at index 0, most recent at
                the last REAL index before padding begins). Padding
                positions contain zero vectors.
            padding_mask: shape (B, max_seq_len), boolean
                True at positions that are PADDING (should be ignored),
                False at positions that are REAL items.
                (This naming convention -- True means "mask this out" --
                is what PyTorch's transformer expects; it can feel
                backwards at first since True usually means "yes, keep it"
                in everyday code, so this is a common point of confusion.)

        Returns:
            Tensor of shape (B, embedding_dim): one user embedding per user,
            L2-normalized.
        """
        batch_size, seq_len, _ = history_embeddings.shape

        # ── Step 1: Add positional embeddings ──────────────────────────────
        # Create position indices [0, 1, 2, ..., seq_len-1] for one sequence,
        # then repeat that same index pattern for every user in the batch.
        #
        # torch.arange(seq_len) → tensor([0, 1, 2, ..., 49])
        # .unsqueeze(0) → adds a batch dimension: shape becomes (1, 50)
        # .expand(batch_size, -1) → repeats it across the batch dimension
        #   without actually copying memory: shape becomes (B, 50)
        #   (-1 means "keep this dimension's size as-is")
        position_ids = torch.arange(seq_len, device=history_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # position_ids shape: (B, 50), every row is [0, 1, 2, ..., 49]

        position_embeds = self.position_embedding(position_ids)
        # position_embeds shape: (B, 50, 128) -- looked up from the embedding table

        # Element-wise addition: each item embedding gets its corresponding
        # positional embedding added directly to it. This is the standard
        # transformer convention (used in BERT/GPT too) -- not concatenation,
        # ADDITION. The model has to learn to "untangle" position information
        # and content information from the combined vector, which empirically
        # works very well and keeps the dimensionality unchanged (still 128).
        history_with_position = history_embeddings + position_embeds
        # shape: (B, 50, 128)

        # ── Step 2: Build the causal mask ──────────────────────────────────
        # PyTorch provides a built-in helper specifically for this:
        # generate_square_subsequent_mask(seq_len) returns a (seq_len, seq_len)
        # matrix where position i is allowed to attend to positions 0..i,
        # and forbidden (-inf) from attending to positions i+1 onward.
        #
        # WHY -inf and not just 0 or False?
        # This mask gets ADDED to the raw attention scores before the
        # softmax step inside the transformer. Adding -infinity to a score
        # forces softmax to assign that position a probability of exactly
        # zero (since e^(-inf) = 0) -- mathematically guaranteeing zero
        # attention weight on forbidden positions, rather than just a small
        # number that could still leak a little signal through.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            history_embeddings.device
        )
        # causal_mask shape: (50, 50)

        # ── Step 3: Run the transformer encoder ─────────────────────────────
        # NOTE: PyTorch expects mask and src_key_padding_mask to be the SAME
        # TYPE (both boolean or both float) to avoid an internal type-mismatch
        # warning. causal_mask is float (0 / -inf, from generate_square_subsequent_mask),
        # so we convert the boolean padding_mask into that same float style here:
        # True (padding) → -inf (forbidden), False (real item) → 0.0 (allowed).
        # This produces IDENTICAL masking behavior to before -- we're only
        # changing the data type used to represent it, not the logic.
        padding_mask_float = torch.zeros_like(padding_mask, dtype=torch.float32)
        padding_mask_float = padding_mask_float.masked_fill(padding_mask, float("-inf"))

        # src_key_padding_mask handles the PADDING mask (ignore pad positions
        #   entirely, for any user whose real history is shorter than 50)
        # mask handles the CAUSAL mask (no peeking at future positions)
        # Both masks apply simultaneously -- a position must be both a real
        # item AND not in the future to be attended to.
        encoded = self.transformer_encoder(
            history_with_position,
            mask=causal_mask,
            src_key_padding_mask=padding_mask_float,
        )
        # encoded shape: (B, 50, 128) -- one contextualized representation
        # per position, for every position in every user's sequence

        # ── Step 4: Extract the user embedding ──────────────────────────────
        # We only care about ONE output per user: the representation at the
        # LAST REAL (non-padding) position in their sequence. This is the
        # representation that has "seen" the user's entire real history
        # (thanks to causal attention) and represents "what this user looks
        # like right now, ready to predict their next interaction."
        #
        # We can't just always take index -1 (the literal last slot in the
        # padded tensor), because for a user with only 8 real items, slots
        # 8 through 49 are padding -- the real "last item" is at index 7,
        # not index 49. We need to find the last real position PER USER.
        #
        # padding_mask is True at padding positions, False at real positions.
        # (~padding_mask) flips this: True at real positions.
        # .sum(dim=1) counts how many real (True) positions each user has
        #   -- this is just that user's actual history length.
        # Subtracting 1 converts a COUNT into the correct zero-indexed
        #   position of the last real item.
        real_lengths = (~padding_mask).sum(dim=1)        # shape: (B,)
        last_real_position = real_lengths - 1              # shape: (B,)

        # Now we gather exactly one timestep per user out of the (B, 50, 128)
        # tensor, using a different index for each user in the batch.
        # torch.arange(batch_size) gives us [0, 1, 2, ..., B-1] to pick the
        # right user row, and last_real_position gives the right timestep
        # for that specific user -- together they index encoded as
        # encoded[user_i, last_real_position_for_user_i, :] for every user
        # simultaneously, without writing an explicit Python for-loop.
        batch_indices = torch.arange(batch_size, device=encoded.device)
        user_embeddings = encoded[batch_indices, last_real_position]
        # user_embeddings shape: (B, 128)

        # ── Step 5: L2 Normalize ─────────────────────────────────────────────
        # Same reasoning as the Item Tower: dot product scoring needs
        # comparable vector magnitudes, so we normalize to unit length.
        normalized = nn.functional.normalize(user_embeddings, p=2, dim=1)

        return normalized
