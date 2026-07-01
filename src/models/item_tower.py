"""
models/item_tower.py

THE ITEM TOWER
==============
Takes a product's image and text (title + description) and produces
a single 128-dimensional embedding vector that represents that product
in a shared embedding space.

INTERVIEW SUMMARY (what to say when asked "explain your item tower"):
  "We use a two-encoder approach: CLIP processes the product image into
   a 512-dim visual embedding, and a Sentence Transformer processes the
   title and description into a 384-dim text embedding. Both encoders are
   frozen -- we use them as fixed feature extractors, not fine-tune them,
   because our dataset is too small to improve pretrained representations
   without catastrophic forgetting. A trainable 3-layer MLP fusion layer
   takes the concatenated 896-dim vector and projects it down to a shared
   128-dim embedding space. This design lets us represent new items with
   zero interaction history (cold-start) purely from their visual and
   textual content."
"""

import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer


# ── Constants ──────────────────────────────────────────────────────────────────

# CLIP produces 512-dimensional image embeddings.
# This is a fixed property of the clip-vit-base-patch32 model.
CLIP_IMAGE_DIM = 512

# Sentence Transformer (all-MiniLM-L6-v2) produces 384-dimensional text embeddings.
# This is a fixed property of that specific model.
SENTENCE_DIM = 384

# The dimension of the final fused item embedding.
# We choose 128 because:
#   - Small enough to be memory-efficient when storing embeddings for millions of items
#   - Large enough to capture meaningful product representations
#   - Matches the user tower output dim so dot-product scoring works directly
EMBEDDING_DIM = 128


# ── Item Tower Class ────────────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """
    Multimodal item encoder: image + text → 128-dim embedding.

    WHY nn.Module?
    PyTorch organizes all neural network components as subclasses of nn.Module.
    This gives us automatic parameter tracking (so the optimizer knows which
    weights to update), the ability to call .train() and .eval() to switch
    between training and inference modes, and .to(device) to move the model
    to GPU when available.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, embedding_cache: dict = None):
        # Always call the parent class constructor first in PyTorch.
        # This initializes the internal bookkeeping nn.Module needs.
        super(ItemTower, self).__init__()

        self.embedding_dim = embedding_dim

        # ── Optional precomputed embedding cache ──────────────────────────
        # If provided, this is the dict produced by precompute_item_embeddings.py:
        #   { item_id: {"image_embedding": Tensor(512), "text_embedding": Tensor(384)} }
        # When an item_id is found in this cache, we SKIP downloading the
        # image and SKIP running CLIP/SentenceTransformer entirely -- we
        # just look up the already-computed raw embeddings. This is purely
        # a speed optimization; the cached values are mathematically
        # IDENTICAL to what CLIP/SentenceTransformer would produce live,
        # since those encoders are frozen and therefore deterministic.
        #
        # If embedding_cache is None (the default), the ItemTower behaves
        # exactly as before -- this change is fully backward compatible
        # with all our existing tests.
        self.embedding_cache = embedding_cache

        # ── Encoder 1: CLIP Vision Encoder (FROZEN) ────────────────────────
        # CLIPModel contains both an image encoder and a text encoder.
        # We only use the image encoder here.
        # "openai/clip-vit-base-patch32" means:
        #   - Trained by OpenAI
        #   - ViT = Vision Transformer architecture (not a CNN)
        #   - base = medium-sized model (not tiny, not huge)
        #   - patch32 = divides the image into 32x32 pixel patches
        #     (ViTs process images as sequences of patches, similar to how
        #      transformers process sequences of tokens in NLP)
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # FREEZE the CLIP weights.
        # requires_grad=False tells PyTorch: "do not compute gradients for
        # these parameters during backpropagation." This means the optimizer
        # will never update CLIP's weights -- it's permanently fixed.
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ── Encoder 2: Sentence Transformer (FROZEN) ───────────────────────
        # all-MiniLM-L6-v2 is a lightweight but effective model for encoding
        # sentences and short paragraphs into dense vectors.
        # "MiniLM" = distilled/compressed version of a larger model
        # "L6" = 6 transformer layers (lightweight)
        # "v2" = second version (improved training)
        print("Loading Sentence Transformer...")
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Freeze the sentence transformer weights too.
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ── Encoder 3: Fusion MLP (TRAINABLE) ─────────────────────────────
        # This is the only part we actually train.
        # Input: concatenated image embedding (512) + text embedding (384) = 896 dims
        # Output: fused item embedding (128 dims)
        #
        # Architecture: 3 layers
        #   Layer 1: 896 → 512 (compress)
        #   Layer 2: 512 → 256 (compress further)
        #   Layer 3: 256 → 128 (final embedding)
        #
        # Between each linear layer we apply:
        #   - BatchNorm1d: normalizes activations to have mean~0, std~1.
        #     This stabilizes training and lets us use higher learning rates.
        #   - ReLU: the activation function. Sets all negative values to 0.
        #     Without a nonlinear activation, stacking linear layers is
        #     mathematically identical to one linear layer -- you'd never
        #     gain expressive power from depth. ReLU is the standard choice.
        #   - Dropout(0.2): randomly zeros 20% of neurons during training.
        #     This prevents overfitting by forcing the network not to rely
        #     too heavily on any single neuron.
        self.fusion_mlp = nn.Sequential(
            # Layer 1
            nn.Linear(CLIP_IMAGE_DIM + SENTENCE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3 (output layer -- no activation here because we want
            # raw values that can be positive or negative; L2 normalization
            # happens after this in the forward pass)
            nn.Linear(256, embedding_dim),
        )

    def _encode_image(self, image_url: str) -> torch.Tensor:
        """
        Downloads an image from a URL and runs it through CLIP's vision encoder.
        Returns a 512-dimensional tensor.

        WHY a separate method?
        Keeping each encoding step in its own method makes the code easier
        to test, debug, and replace. If we later want to swap CLIP for a
        different vision encoder, we only change this one method.
        """
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            # If the image fails to load (broken URL, network issue),
            # return a zero vector rather than crashing. This is important
            # for robustness -- real datasets always have some broken URLs.
            return torch.zeros(CLIP_IMAGE_DIM)

        # CLIPProcessor handles all the preprocessing CLIP expects:
        # resizing to 224x224, normalizing pixel values, converting to tensor.
        # return_tensors="pt" means "return PyTorch tensors" (not numpy).
        inputs = self.clip_processor(images=image, return_tensors="pt")

        # torch.no_grad() tells PyTorch: "don't track operations for
        # gradient computation inside this block." Since CLIP is frozen,
        # we never need gradients through it, so this saves memory and
        # speeds up the forward pass.
        with torch.no_grad():
            # get_image_features() runs only the vision encoder part of CLIP,
            # skipping the text encoder entirely.
            image_features = self.clip_model.get_image_features(**inputs)

        # image_features shape: (1, 512) -- batch size 1, 512 dimensions
        # We squeeze out the batch dimension to get shape (512,)
        return image_features.squeeze(0)

    def _encode_text(self, title: str, description: str) -> torch.Tensor:
        """
        Concatenates title and description, encodes with Sentence Transformer.
        Returns a 384-dimensional tensor.

        WHY concatenate title and description?
        The title is short and specific ("Women's Running Shoes").
        The description is long and rich ("Lightweight mesh upper for breathability...").
        Together they give the text encoder the most complete picture of the product.
        We separate them with a period so the encoder treats them as related but
        distinct sentences rather than one run-on string.
        """
        text = f"{title}. {description}".strip()

        # encode() returns a numpy array by default.
        # convert_to_tensor=True gives us a PyTorch tensor directly.
        with torch.no_grad():
            text_features = self.text_encoder.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # text_features shape: (384,)
        return text_features

    def forward(
        self,
        image_urls: list[str],
        titles: list[str],
        descriptions: list[str],
        item_ids: list = None,
    ) -> torch.Tensor:
        """
        The forward pass: given a batch of items, return their embeddings.

        WHY batches and not single items?
        Neural networks are much more efficient when processing multiple
        examples simultaneously (a "batch") rather than one at a time.
        Modern GPUs are designed for parallel matrix operations -- feeding
        them one item at a time wastes most of their capacity.

        Args:
            image_urls:   list of B image URLs (B = batch size)
            titles:       list of B product titles
            descriptions: list of B product descriptions
            item_ids:     OPTIONAL list of B item IDs. If provided AND
                self.embedding_cache is set, we look up precomputed raw
                embeddings instead of downloading images / running the
                encoders live -- a pure speed optimization with NO effect
                on the mathematical result (see precompute_item_embeddings.py
                for why these cached values are exactly equivalent).
                If item_ids is None (the default), behavior is UNCHANGED
                from before -- every item is encoded live, exactly as our
                existing tests already verified.

        Returns:
            Tensor of shape (B, 128): one L2-normalized embedding per item
        """
        use_cache = self.embedding_cache is not None and item_ids is not None

        # ── Step 1: Get image embeddings (cached or live) ───────────────────
        if use_cache:
            image_embeddings = torch.stack([
                self.embedding_cache[iid]["image_embedding"] for iid in item_ids
            ])
        else:
            # Original live path -- unchanged, still used whenever no cache
            # / no item_ids are supplied (e.g. all our existing tests).
            image_embeddings = torch.stack([
                self._encode_image(url) for url in image_urls
            ])
        # image_embeddings shape: (B, 512)

        # ── Step 2: Get text embeddings (cached or live) ────────────────────
        if use_cache:
            text_embeddings = torch.stack([
                self.embedding_cache[iid]["text_embedding"] for iid in item_ids
            ])
        else:
            text_embeddings = torch.stack([
                self._encode_text(t, d) for t, d in zip(titles, descriptions)
            ])
        # text_embeddings shape: (B, 384)

        # ── Step 3: Concatenate image and text embeddings ──────────────────
        # torch.cat joins tensors along a specified dimension.
        # dim=1 means we concatenate along the feature dimension (not the batch dimension).
        # (B, 512) + (B, 384) → (B, 896)
        combined = torch.cat([image_embeddings, text_embeddings], dim=1)
        # combined shape: (B, 896)

        # ── Step 4: Pass through Fusion MLP ───────────────────────────────
        # NOTE: this step is NEVER cached, and never will be -- the fusion
        # MLP's weights change during training, so its output must be
        # recomputed fresh on every forward pass for gradients to flow
        # correctly and for the output to reflect the model's current
        # (improving) state.
        fused = self.fusion_mlp(combined)
        # fused shape: (B, 128)

        # ── Step 5: L2 Normalize the output ───────────────────────────────
        # L2 normalization scales each embedding vector so its length is exactly 1.
        # WHY? Because our scoring function is a dot product. If embeddings have
        # different lengths (magnitudes), a long vector will always score higher
        # than a short one regardless of direction -- which means the model could
        # "cheat" by just making its embeddings very large rather than learning
        # meaningful directions. L2 normalization removes the magnitude, so the
        # dot product becomes equivalent to cosine similarity -- purely about
        # the direction (semantic meaning) of the vectors.
        normalized = nn.functional.normalize(fused, p=2, dim=1)
        # normalized shape: (B, 128), every row has L2 norm = 1.0

        return normalized
