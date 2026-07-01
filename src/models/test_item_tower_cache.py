"""
test_item_tower_cache.py

Proves that ItemTower's cache path produces IDENTICAL output to the live
(non-cached) path, for the same item. This is the core correctness
guarantee behind the whole caching optimization: caching must be
completely invisible to the model's behavior -- it should only affect
speed, never the actual numbers produced.

Run on Discovery:
    cd ~/fed_multimodal_rec/src/models
    python test_item_tower_cache.py
"""

import torch
from item_tower import ItemTower


def test_cached_path_matches_live_path():
    image_url = "https://m.media-amazon.com/images/I/61c5j1R6+aL._AC_SX679_.jpg"
    title = "Blue Cotton T-Shirt"
    description = "A comfortable everyday t-shirt made from 100% cotton."
    item_id = "fake_item_001"

    # ── Step 1: compute the embedding LIVE (no cache) ───────────────────────
    tower_live = ItemTower()
    tower_live.eval()

    with torch.no_grad():
        live_output = tower_live(
            image_urls=[image_url],
            titles=[title],
            descriptions=[description],
            # item_ids deliberately omitted -- forces the live path,
            # exactly as all our previous tests already did
        )

    # ── Step 2: manually build a cache containing this item's RAW embeddings ─
    # In real usage, precompute_item_embeddings.py would generate this by
    # running the same encoders once. Here we extract them directly from
    # tower_live's own encoders to build a matching cache entry, so we're
    # testing the CACHE LOOKUP + FUSION logic specifically, not re-testing
    # whether CLIP/SentenceTransformer are deterministic (we already know
    # they are, since they're frozen).
    with torch.no_grad():
        raw_image_embedding = tower_live._encode_image(image_url)
        raw_text_embedding = tower_live._encode_text(title, description)

    fake_cache = {
        item_id: {
            "image_embedding": raw_image_embedding,
            "text_embedding": raw_text_embedding,
        }
    }

    # ── Step 3: compute the embedding via the CACHE path ────────────────────
    # NOTE: we build a SEPARATE ItemTower instance here on purpose. If we
    # reused tower_live, we couldn't be fully sure the cache path wasn't
    # secretly just falling through to results already computed for the
    # live path moments earlier. A fresh instance with its OWN freshly
    # initialized fusion_mlp ensures we're testing the cache mechanism
    # honestly -- though note the fusion_mlp weights are randomly
    # initialized per-instance, so for this comparison to be meaningful we
    # actually need the SAME fusion_mlp weights in both towers. We handle
    # that by copying the state dict across, isolating the test to ONLY
    # the cache-vs-live difference, nothing else.
    tower_cached = ItemTower(embedding_cache=fake_cache)
    tower_cached.load_state_dict(tower_live.state_dict())
    tower_cached.eval()

    with torch.no_grad():
        cached_output = tower_cached(
            image_urls=[image_url],   # still passed, but IGNORED when cache hits
            titles=[title],            # same -- ignored when cache hits
            descriptions=[description],
            item_ids=[item_id],
        )

    # ── Step 4: compare ──────────────────────────────────────────────────────
    assert torch.allclose(live_output, cached_output, atol=1e-6), (
        f"Cached and live outputs differ! This means the cache is NOT a "
        f"pure speed optimization -- it's silently changing results.\n"
        f"Live:   {live_output}\n"
        f"Cached: {cached_output}"
    )
    print("PASS: cached path produces IDENTICAL output to live computation")
    print(f"      (max absolute difference: {(live_output - cached_output).abs().max().item():.2e})")


if __name__ == "__main__":
    print("=" * 60)
    print("ItemTower Cache Correctness Test")
    print("=" * 60)

    test_cached_path_matches_live_path()

    print()
    print("=" * 60)
    print("Cache correctness verified.")
    print("=" * 60)
