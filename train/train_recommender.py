# train_recommender.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import os
import argparse
from tqdm import tqdm

from models import DiffusionRecommenderModel
from diffusion import DiffusionScheduler
from dataset import get_dataset
from evaluate import evaluate_model
from utils import save_checkpoint, set_seed

def train_recommender(args):
    set_seed(args.seed)

    train_dataset, val_dataset, tokenizer = get_dataset(args.dataset, args, stage="train_recommender")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = DiffusionRecommenderModel(args, tokenizer).to(args.device)
    scheduler = DiffusionScheduler(args)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ndcg = 0.0
    step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            output = model(
                user_history=batch['history_ids'],
                identifier=batch['target_ids'],
                diffusion_step=batch['timestep'],
                attention_mask=batch['attention_mask'],
                metadata=batch['metadata'],
                set_labels=batch['target_set']
            )

            total_loss = output['total_loss']
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=total_loss.item())

            if step % args.eval_steps == 0:
                model.eval()
                metrics = evaluate_model(model, val_loader, tokenizer, args)
                ndcg = metrics['NDCG@10']
                print(f"[Eval @ Step {step}] NDCG@10 = {ndcg:.4f}, Recall@10 = {metrics['Recall@10']:.4f}")
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    save_checkpoint(model, args.save_dir, f"best_model.pt")

        save_checkpoint(model, args.save_dir, f"epoch{epoch}_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazonbook")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_recommender(args)