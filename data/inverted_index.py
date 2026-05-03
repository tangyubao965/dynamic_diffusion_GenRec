import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


class InvertedIndex:
    def __init__(self, min_freq=1):
        self.index = defaultdict(set)
        self.min_freq = min_freq
        self.token_freq = defaultdict(int)

    def build(self, data_path, field="metadata.title"):
        with open(data_path, 'r') as f:
            for line in tqdm(f, desc=f'Building Inverted Index from {field}'):
                item = json.loads(line)
                target = item['target']
                metadata = item['metadata']
                value = self._extract_field(metadata, field)
                tokens = self._tokenize(value)
                for tok in tokens:
                    self.token_freq[tok] += 1

        with open(data_path, 'r') as f:
            for line in tqdm(f, desc='Populating Index'):
                item = json.loads(line)
                item_id = item['target']
                metadata = item['metadata']
                value = self._extract_field(metadata, field)
                tokens = self._tokenize(value)
                for tok in tokens:
                    if self.token_freq[tok] >= self.min_freq:
                        self.index[tok].add(item_id)

        self.index = {tok: list(ids) for tok, ids in self.index.items()}

    def _extract_field(self, metadata, field_path):
        keys = field_path.split(".")
        val = metadata
        for k in keys:
            val = val.get(k, "")
            if not isinstance(val, dict):
                break
        return val if isinstance(val, str) else ""

    def _tokenize(self, text):
        return text.lower().replace(",", " ").replace(".", " ").split()

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            self.index = json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--index_path', type=str, required=True, help='Where to save index')
    parser.add_argument('--field', type=str, default='metadata.title', help='Field to index')
    parser.add_argument('--min_freq', type=int, default=1)
    args = parser.parse_args()

    index = InvertedIndex(min_freq=args.min_freq)
    index.build(args.data_path, field=args.field)
    index.save(args.index_path)