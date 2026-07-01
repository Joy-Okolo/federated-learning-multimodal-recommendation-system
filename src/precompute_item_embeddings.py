"""
precompute_item_embeddings.py

PURPOSE:
Computes the FROZEN parts of each item's representation -- the raw CLIP
image embedding (512-dim) and raw Sentence Transformer text embedding
(384-dim) -- ONE TIME for every item in the catalog, and saves them to disk.

WHY THIS EXISTS:
CLIP and the Sentence Transformer never change during training (they're
frozen). Recomputing their outputs for the same item, over and over, across
every training epoch, is pure wasted computation. This script front-loads
that one-time cost so the training loop can just look up a cached vector
instantly instead of downloading an image and running it through CLIP
every single time that item appears in a batch.

WHAT GETS CACHED (and what doesn't):
  CACHED: raw CLIP image embedding, raw Sentence Transformer text embedding
  NOT CACHED: the fusion MLP's output -- that depends on the fusion MLP's
              CURRENT weights, which change during training, so it must be
              recomputed fresh every forward pass.

USAGE:
    python precompute_item_embeddings.py \
        --items_parquet ../../data/items.parquet \
        --out_cache ../../data/item_embedding_cache.pt
"""

import argparse

import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer


def load_image_embedding(clip_model, clip_processor, image_url: str) -> torch.Tensor:
    """
    Same logic as ItemTower._encode_image, duplicated here deliberately
    (rather than importing ItemTower) because this script's job is to
    produce RAW encoder outputs only -- it has no reason to instantiate
    the full Recommender/ItemTower class, which also builds the fusion MLP
    we explicitly do NOT want to use here.
    """
    try:
        response = requests.get(image_url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return torch.zeros(512)

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_parquet", type=str, default="../../data/items.parquet")
    parser.add_argument("--out_cache", type=str, default="../../data/item_embedding_cache.pt")
    args = parser.parse_args()

    print(f"Loading items from {args.items_parquet}...")
    items_df = pd.read_parquet(args.items_parquet)
    print(f"Found {len(items_df):,} items to encode")

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()  # we never train this, so eval mode throughout

    print("Loading Sentence Transformer...")
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # The cache itself: a dictionary mapping item_id -> (image_embedding, text_embedding)
    # We use a plain Python dict + torch.save rather than a database, since
    # for a few tens of thousands of items this comfortably fits in memory
    # and on disk as a single file. If the catalog grows to millions of
    # items, this would need to move to a proper key-value store (e.g.
    # LMDB or a vector database) -- worth mentioning as a scaling
    # consideration in an interview, even though it's not needed yet here.
    cache = {}

    print("Encoding items (this is the one-time cost)...")
    for _, row in tqdm(items_df.iterrows(), total=len(items_df)):
        item_id = row["item_id"]

        image_embedding = load_image_embedding(clip_model, clip_processor, row["image_url"])

        text = f"{row['title']}. {row['description']}".strip()
        with torch.no_grad():
            text_embedding = text_encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)

        cache[item_id] = {
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
        }

    print(f"Saving cache for {len(cache):,} items to {args.out_cache}...")
    torch.save(cache, args.out_cache)
    print("Done.")

    # Quick sanity check on the saved file
    reloaded = torch.load(args.out_cache)
    sample_item_id = next(iter(reloaded))
    sample = reloaded[sample_item_id]
    print(f"\nSanity check -- sample item '{sample_item_id}':")
    print(f"  image_embedding shape: {sample['image_embedding'].shape}")
    print(f"  text_embedding shape:  {sample['text_embedding'].shape}")


if __name__ == "__main__":
    main()
