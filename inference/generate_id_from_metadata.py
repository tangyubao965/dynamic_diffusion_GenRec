import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from models.models import DiffusionRecommenderModel
from models.diffusion import DiffusionScheduler
from datasets.metadata_dataset import MetadataDataset  # dataset that only returns metadata input
from utils.io import load_tokenizer, load_model_checkpoint, save_generation_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=5)  # number of generations per item
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

@torch.no_grad()
def generate_identifiers(args):
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.model_path)
    model = DiffusionRecommenderModel.from_pretrained(args.model_path, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    diffusion = DiffusionScheduler()

    dataset = MetadataDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_results = []

    for batch in tqdm(dataloader, desc="Generating identifiers"):
        metadata_input = batch['metadata_input'].to(device)  # tokenized metadata
        attention_mask = batch['attention_mask'].to(device)
        item_ids = batch['item_id']

        batch_results = []
        for _ in range(args.num_samples):
            noise = diffusion.sample_noise(metadata_input.shape)
            gen_ids = model.generate_from_metadata(metadata_input, attention_mask, noise=noise, tokenizer=tokenizer)
            batch_results.append(gen_ids)

        # batch_results: [num_samples][batch_size][token_ids]
        batch_results = list(zip(*batch_results))  # [batch_size][num_samples][token_ids]

        for i, item_id in enumerate(item_ids):
            decoded_list = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_results[i]]
            all_results.append({
                "item_id": item_id,
                "generated_ids": decoded_list
            })

    save_generation_results(all_results, args.output_path)

if __name__ == '__main__':
    args = parse_args()
    generate_identifiers(args)