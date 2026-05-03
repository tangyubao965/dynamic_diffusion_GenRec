# DDGR: Diffusion-based Generative Recommendation

This repository provides the implementation of **DDGR**, a diffusion-based generative recommendation framework. DDGR formulates item semantic identifier (SID) generation as conditional denoising in a continuous latent space. It supports hybrid SID grounding, prefix-relaxed recommendation generation, and recommendation-aware SID evolution.

## Overview

DDGR contains three main stages:

1. **SID grounding**  
   Initialize item SIDs from hybrid item-side context, including textual metadata and collaborative signals.

2. **Recommendation training**  
   Train a user-conditioned denoising recommender to generate the SID of the next item from user history.

3. **SID evolution**  
   Regenerate item SIDs using the trained recommender and fine-tune the recommendation model on the updated SID space.

During inference, DDGR generates token distributions over the full SID and ranks valid items with an order-robust SID score.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The code is implemented in Python and PyTorch. We recommend using a CUDA-enabled GPU for training.

## Data Preparation

Place the preprocessed datasets under the `data/` directory. Each dataset should contain user interaction sequences and item metadata.

A typical dataset should include:

```text
train.txt
valid.txt
test.txt
item_metadata.json
```

where:

- `train.txt`, `valid.txt`, and `test.txt` store user-item interaction sequences.
- `item_metadata.json` stores item-side textual fields such as title, category, brand, author, or description.

To preprocess raw data, run:

```bash
python utils/preprocess.py \
  --dataset amazon_book \
  --input_dir /path/to/raw/data \
  --output_dir data/amazon_book
```

## Training

DDGR is trained in three stages.

### 1. Train SID Grounding Model

This stage learns initial item SIDs from item-side context.

```bash
python trainer/train_identifier.py \
  --config config/amazon_book.yaml
```

After training, generate initial SIDs for all items:

```bash
python inference/generate_from_metadata.py \
  --config config/amazon_book.yaml \
  --checkpoint checkpoints/amazon_book/identifier.pt \
  --output_path data/amazon_book/initial_sids.json
```

### 2. Train Recommendation Model

This stage trains the user-conditioned denoising recommender using the initialized SID space.

```bash
python trainer/train_recommender.py \
  --config config/amazon_book.yaml \
  --sid_path data/amazon_book/initial_sids.json
```

### 3. Refine Item SIDs

This stage performs recommendation-aware SID evolution.

```bash
python trainer/refine_identifier.py \
  --config config/amazon_book.yaml \
  --recommender_checkpoint checkpoints/amazon_book/recommender.pt \
  --sid_path data/amazon_book/initial_sids.json \
  --output_path data/amazon_book/refined_sids.json
```

The refined SIDs can then be used to further fine-tune the recommendation model:

```bash
python trainer/train_recommender.py \
  --config config/amazon_book.yaml \
  --sid_path data/amazon_book/refined_sids.json \
  --resume checkpoints/amazon_book/recommender.pt
```

## End-to-End Training

You can also run the full pipeline through the entry script:

```bash
python main.py \
  --config config/amazon_book.yaml \
  --mode train
```

The full pipeline includes SID grounding, recommendation training, SID evolution, and final model selection.

## Inference

To generate recommendations from user histories, run:

```bash
python inference/generate_from_history.py \
  --config config/amazon_book.yaml \
  --checkpoint checkpoints/amazon_book/recommender_final.pt \
  --sid_path data/amazon_book/refined_sids.json \
  --output_path results/amazon_book_predictions.json
```

During inference, the final evolved SIDs are used to build a token-to-item inverted index. DDGR first produces token distributions over the SID vocabulary, then selects candidate items according to SID token overlap, and finally ranks them with an order-robust SID score.

## Configuration

All major hyperparameters are specified in YAML configuration files under `config/`.

Important options include:

```yaml
dataset: amazon_book
sid_length: 10
hidden_size: 512
num_layers: 6
num_heads: 8
ffn_dim: 2048
dropout: 0.1

diffusion_steps: 100
noise_schedule: linear

batch_size: 256
learning_rate: 1e-4
weight_decay: 1e-4

mask_ratio: 0.3
lambda_mask: 0.1
lambda_set_ground: 0.1
lambda_set_rec: 0.1

evolution_rounds: 3
top_b_tokens: 20
top_m_candidates: 1000
```

Dataset-specific settings can be adjusted in the corresponding config file.

## Evaluation

The default evaluation follows the leave-one-out sequential recommendation protocol. The last interaction of each user is used for testing, the second last interaction is used for validation, and the remaining interactions are used for training.

We report:

- Recall@10
- Recall@20
- NDCG@10
- NDCG@20

To evaluate a trained model:

```bash
python main.py \
  --config config/amazon_book.yaml \
  --mode evaluate \
  --checkpoint checkpoints/amazon_book/recommender_final.pt \
  --sid_path data/amazon_book/refined_sids.json
```

## Reproducibility

We recommend setting a random seed before training:

```bash
python main.py \
  --config config/amazon_book.yaml \
  --mode train \
  --seed 2025
```

For stable results, run each experiment multiple times with different random seeds and report the average performance.

## License

This repository is released for research purposes.
