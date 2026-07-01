# Federated Multimodal E-Commerce Recommender

A privacy-preserving, multimodal recommendation system for e-commerce,
combining **vision-language item representations** with **federated learning**
to enable personalized recommendations without centralizing user data.

Built as part of a research portfolio targeting Applied Scientist roles,
this project bridges existing federated learning research (CLAMP, MA-CLAMP,
MOSAIC) with LLM-era multimodal representation learning.

---

## What This Project Does

Given a user's chronological purchase/review history, the system predicts
the **next item** the user is likely to interact with, ranked from a catalog
of candidates.

**What makes it different from a standard recommender:**

1. **Multimodal item representations** -- each product is encoded using both
   its image (via frozen CLIP vision encoder) and its text metadata (title +
   description, via frozen Sentence Transformer). This enables meaningful
   recommendations for cold-start items with no interaction history.

2. **Federated training** -- users are partitioned into simulated clients
   by activity level, mimicking real edge-device heterogeneity, and model
   weights are aggregated via FedAvg rather than sharing raw user data.

3. **General e-commerce schema** -- platform-agnostic: Amazon Reviews 2023
   is used as a public benchmark, but any e-commerce dataset plugs in via
   a simple adapter interface.

---

## Architecture
ITEM TOWER                          USER TOWER
──────────────────────────────      ───────────────────────────────
[Product Image]                     For each item in history:
↓                                   encode via Item Tower
[CLIP Vision Encoder] (frozen)
↓ 512-dim                       [item_emb_1, ..., item_emb_n]
[image embedding] ──┐                   + positional embeddings
├→ Fusion MLP        ↓
[text embedding]  ──┘ → 128-dim     [Transformer Encoder]
↓
[Title + Description]               [user embedding] (128-dim)
↓
[Sentence Transformer] (frozen)
↓ 384-dim                       SCORING
[text embedding]                    score = dot_product(user_emb, item_emb)

**Key design decisions:**
- CLIP and Sentence Transformer are **frozen** (transfer learning)
- A **single shared ItemTower** encodes both history and candidate items
- Causal masking in the User Tower prevents looking ahead in the sequence
- **Pre-computed embedding cache** avoids redundant frozen encoder passes

---

## Training

- **Task:** Next-item prediction (sequential recommendation)
- **Split:** Leave-one-out evaluation
- **Loss:** Sampled softmax cross-entropy (1 positive + 99 negatives)
- **Metrics:** Hit Rate @ K and NDCG @ K (K = 5, 10)
- **Federated:** FedAvg across clients partitioned by activity level

---

## Dataset

Amazon Reviews 2023 (McAuley Lab), Clothing/Shoes/Jewelry category.
Subsample: 5,000 users, 56,154 unique items, 72,231 interactions.

---
## Project Structure
fed_multimodal_rec/
├── src/
│   ├── schema.py                      # generic e-commerce data contract
│   ├── adapters/
│   │   └── amazon_reviews_adapter.py  # Amazon Reviews 2023 → generic schema
│   ├── models/
│   │   ├── item_tower.py              # multimodal product encoder (CLIP + SentenceTransformer)
│   │   ├── user_tower.py              # causal transformer sequence encoder
│   │   └── recommender.py             # two-tower model with dot-product scoring
│   ├── training/
│   │   ├── dataset.py                 # leave-one-out splits + negative sampling
│   │   ├── collate.py                 # custom DataLoader collate function
│   │   ├── loss.py                    # sampled softmax cross-entropy
│   │   └── train.py                   # full training loop
│   └── precompute_item_embeddings.py  # one-time frozen encoder caching
├── slurm/                             # SLURM batch scripts for HPC training
├── requirements.txt
└── README.md

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/fed-multimodal-rec.git
cd fed-multimodal-rec
pip install -r requirements.txt

# Pull and subsample real data
cd src/adapters
python amazon_reviews_adapter.py --min_interactions 5 --max_users 5000

# Pre-compute frozen item embeddings (run once)
cd src
python precompute_item_embeddings.py

# Train
cd src/training
python train.py --epochs 5 --batch_size 16 --lr 1e-4
```

---

## Related Work

- **CLAMP** (ACM SAC 2026): adaptive model pruning for stragglers in federated edge computing
- **MA-CLAMP** (ACM SIGAPP ACR 2026): mask-aware aggregation for heterogeneous federated settings
- **MOSAIC** (dissertation, in progress): heterogeneity-aware multimodal federated learning for IoT/edge AI

---

## Author

Joy Okolo -- PhD Candidate, Computer Science, South Dakota State University
[LinkedIn](https://www.linkedin.com/in/joy-okolo/) | joy.okolo@jacks.sdstate.edu
