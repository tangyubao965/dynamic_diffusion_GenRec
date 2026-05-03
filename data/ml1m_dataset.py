import json
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_history_len=50, max_id_len=10, prompt_mask_ratio=0.3, task='rec'):
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_id_len = max_id_len
        self.prompt_mask_ratio = prompt_mask_ratio
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Metadata fields: title, genres
        metadata = item['metadata']
        meta_text = f"{metadata.get('title', '')}, {metadata.get('genres', '')}"
        tokens = self.tokenizer(meta_text, max_length=self.max_id_len, padding='max_length', truncation=True, return_tensors='pt')

        if self.task == 'id':
            input_ids = tokens['input_ids'].squeeze()
            mask = torch.rand(input_ids.shape).lt(self.prompt_mask_ratio)
            input_ids[mask] = self.tokenizer.mask_token_id
            return {
                'input_ids': input_ids,
                'labels': tokens['input_ids'].squeeze()
            }

        history = item['history'][-self.max_history_len:]
        history_tokens = self.tokenizer(history, padding='max_length', truncation=True, max_length=self.max_history_len * self.max_id_len, return_tensors='pt')
        target_tokens = self.tokenizer(item['target'], padding='max_length', truncation=True, max_length=self.max_id_len, return_tensors='pt')

        return {
            'history_input_ids': history_tokens['input_ids'].squeeze(),
            'target_ids': target_tokens['input_ids'].squeeze()
        }