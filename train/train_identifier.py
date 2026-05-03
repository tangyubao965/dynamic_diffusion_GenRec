import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DiffusionRecommenderModel
from diffusion import DiffusionScheduler
from datasets import get_dataset
from utils.losses import diffusion_loss, mask_prediction_loss
from utils.trainer import save_checkpoint, set_seed

import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['amazon', 'yelp', 'ml1m'], required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='checkpoints/')
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(42)

    # Load dataset
    train_dataset = get_dataset(args.dataset, stage='identifier')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Init model & diffusion
    model = DiffusionRecommenderModel(args.dataset).to(args.device)
    diffusion = DiffusionScheduler()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            # batch: dict with keys: metadata_input, target_id, prompt_masked, token_masked

            metadata_input = batch['metadata_input'].to(args.device)                # [B, T]
            prompt_masked_input = batch['prompt_masked'].to(args.device)           # [B, T]
            token_masked_id = batch['token_masked'].to(args.device)                # [B, L]
            target_id = batch['target_id'].to(args.device)                         # [B, L]

            optimizer.zero_grad()

            # Diffusion target reconstruction
            diffusion_input = target_id
            t = diffusion.sample_timesteps(batch_size=metadata_input.size(0))
            noised_id, noise = diffusion.add_noise(diffusion_input, t)
            pred_noise = model(noised_id, metadata_input, timestep=t)
            loss_diffusion = diffusion_loss(pred_noise, noise)

            # Prompt-masked generation (predict full ID from masked metadata)
            pred_from_prompt = model.generate_from_prompt(prompt_masked_input)
            loss_prompt = mask_prediction_loss(pred_from_prompt, target_id)

            # Token-masked generation (predict tokens from corrupted ID)
            pred_from_token_mask = model(token_masked_id, metadata_input, timestep=None)
            loss_token = mask_prediction_loss(pred_from_token_mask, target_id)

            # Total loss
            total_loss = loss_diffusion + 0.5 * (loss_prompt + loss_token)
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=total_loss.item())

        save_checkpoint(model, args.save_path, f'identifier_epoch{epoch+1}.pt')

if __name__ == '__main__':
    train()