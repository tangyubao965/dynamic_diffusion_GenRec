import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from models.models import DiffusionRecommenderModel
from models.diffusion import DiffusionScheduler
from datasets.history_dataset import HistoryRecommendationDataset  # include history + candidates
from utils.io import load_tokenizer, save_ranking_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

@torch.no_grad()
def rank_candidates(args):
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.model_path)
    model = DiffusionRecommenderModel.from_pretrained(args.model_path, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    diffusion = DiffusionScheduler()

    dataset = HistoryRecommendationDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_results = []

    for batch in tqdm(dataloader, desc="Ranking identifiers"):
        history_input = batch['history_input'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        candidate_ids = batch['candidate_ids']  # list of list of token ids
        session_ids = batch['session_id']       # unique identifier for each user session

        batch_size = len(session_ids)
        ranked_batch = []

        for i in range(batch_size):
            history = history_input[i].unsqueeze(0)
            mask = attention_mask[i].unsqueeze(0)

            ranked_candidates = []
            for cand in candidate_ids[i]:
                cand_input = torch.tensor(cand).unsqueeze(0).to(device)
                log_probs = model.compute_token_logprobs(
                    history_input=history,
                    attention_mask=mask,
                    target_identifier=cand_input
                )
                score = log_probs.sum().item()
                decoded = tokenizer.decode(cand, skip_special_tokens=True)
                ranked_candidates.append((decoded, score))

            ranked_candidates.sort(key=lambda x: x[1], reverse=True)
            all_results.append({
                "session_id": session_ids[i],
                "ranked": ranked_candidates
            })

    save_ranking_results(all_results, args.output_path)

if __name__ == '__main__':
    args = parse_args()
    rank_candidates(args)