import argparse
import yaml
import os
import torch
import random
import numpy as np

from train.train_identifier import train_identifier
from train.train_recommender import train_recommender
from train.refine_identifier import refine_identifier
from inference.generate_from_metadata import generate_from_metadata
from inference.generate_from_history import generate_from_history

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--mode', type=str, required=True, choices=[
        'train_identifier', 'train_recommender', 'refine_identifier',
        'inference_metadata', 'inference_recommend'
    ])
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint path')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    if args.mode == 'train_identifier':
        train_identifier(config)

    elif args.mode == 'train_recommender':
        train_recommender(config, checkpoint=args.checkpoint)

    elif args.mode == 'refine_identifier':
        refine_identifier(config, checkpoint=args.checkpoint)

    elif args.mode == 'inference_metadata':
        generate_from_metadata(config, checkpoint=args.checkpoint)

    elif args.mode == 'inference_recommend':
        generate_from_history(config, checkpoint=args.checkpoint)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()