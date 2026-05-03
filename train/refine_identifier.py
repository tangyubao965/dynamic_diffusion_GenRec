import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models import DiffusionRecommenderModel
from diffusion import DiffusionScheduler
from utils import save_checkpoint, load_checkpoint, evaluate_model
from datasets import get_dataset

def refine_identifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and model
    dataset = get_dataset(args.dataset)
    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val'], batch_size=args.batch_size)
    test_loader = DataLoader(dataset['test'], batch_size=args.batch_size)

    model = DiffusionRecommenderModel(args).to(device)
    diffusion = DiffusionScheduler(args).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, "best_recommender.pt")
    load_checkpoint(checkpoint_path, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_ndcg = 0.0
    for round in range(args.num_refine_rounds):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Refinement Round {round+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(batch, mode="refine")  # switch model into refine mode
            diffusion_loss = diffusion(batch, output)
            loss = diffusion_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Round {round+1}, Avg Loss = {total_loss / len(train_loader):.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            recall10, recall20, ndcg10, ndcg20 = evaluate_model(model, test_loader, device)

        print(f"[Eval] R@10: {recall10:.4f}, R@20: {recall20:.4f}, N@10: {ndcg10:.4f}, N@20: {ndcg20:.4f}")
        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            save_checkpoint(model, os.path.join(args.checkpoint_dir, "best_refined.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_refine_rounds", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    refine_identifier(args)