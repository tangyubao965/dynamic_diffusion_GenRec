import os
import json
import argparse
from tqdm import tqdm


def preprocess_amazon(input_path, output_path, max_history_len=50):
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    processed = []
    for user in tqdm(raw_data, desc='Processing Amazon'):
        history = user['history'][-max_history_len:]
        for target in user['targets']:
            item_id = target['item_id']
            metadata = target['metadata']  # e.g., title, authors, publisher
            identifier = f"{metadata.get('title', '')}, {metadata.get('authors', '')}, {metadata.get('publisher', '')}"
            processed.append({
                'history': history,
                'target': identifier,
                'metadata': metadata
            })

    with open(output_path, 'w') as fout:
        for line in processed:
            fout.write(json.dumps(line) + '\n')


def preprocess_yelp(input_path, output_path, max_history_len=50):
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    processed = []
    for user in tqdm(raw_data, desc='Processing Yelp'):
        history = user['history'][-max_history_len:]
        for target in user['targets']:
            metadata = target['metadata']  # e.g., name, categories, address, stars
            identifier = f"{metadata.get('name', '')}, {metadata.get('categories', '')}, {metadata.get('address', '')}, {metadata.get('stars', '')}"
            processed.append({
                'history': history,
                'target': identifier,
                'metadata': metadata
            })

    with open(output_path, 'w') as fout:
        for line in processed:
            fout.write(json.dumps(line) + '\n')


def preprocess_ml1m(input_path, output_path, max_history_len=50):
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    processed = []
    for user in tqdm(raw_data, desc='Processing ML-1M'):
        history = user['history'][-max_history_len:]
        for target in user['targets']:
            metadata = target['metadata']  # e.g., title, genres
            identifier = f"{metadata.get('title', '')}, {metadata.get('genres', '')}"
            processed.append({
                'history': history,
                'target': identifier,
                'metadata': metadata
            })

    with open(output_path, 'w') as fout:
        for line in processed:
            fout.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["amazon", "yelp", "ml1m"])
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_history_len", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.dataset == "amazon":
        preprocess_amazon(args.input_path, args.output_path, args.max_history_len)
    elif args.dataset == "yelp":
        preprocess_yelp(args.input_path, args.output_path, args.max_history_len)
    elif args.dataset == "ml1m":
        preprocess_ml1m(args.input_path, args.output_path, args.max_history_len)