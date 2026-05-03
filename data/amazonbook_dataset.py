import json
import random
import torch
from torch.utils.data import Dataset


class AmazonBookGenRecDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        interaction_path,
        tokenizer,
        stage="id_generation",   # "id_generation" or "recommendation"
        max_text_len=128,
        max_id_len=16,
        max_history_len=20,
        prompt_mask_ratio=0.3,
        mask_pred_ratio=0.15
    ):
        self.stage = stage
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_id_len = max_id_len
        self.max_history_len = max_history_len
        self.prompt_mask_ratio = prompt_mask_ratio
        self.mask_pred_ratio = mask_pred_ratio

        with open(metadata_path, "r") as f:
            self.meta = json.load(f)

        with open(interaction_path, "r") as f:
            self.user_histories = json.load(f)

        self.item_ids = list(self.meta.keys())

        self.samples = []
        if stage == "id_generation":
            for item_id in self.item_ids:
                self.samples.append(("item", item_id))
        elif stage == "recommendation":
            for user, seq in self.user_histories.items():
                if len(seq) < 2:
                    continue
                for i in range(1, len(seq)):
                    hist = seq[max(0, i - max_history_len):i]
                    tgt = seq[i]
                    self.samples.append(("rec", hist, tgt))
        else:
            raise ValueError("stage must be id_generation or recommendation")

    def __len__(self):
        return len(self.samples)

    def _build_metadata_text(self, item_id):
        m = self.meta[item_id]
        parts = []
        if "title" in m and m["title"]:
            parts.append("Title: " + m["title"])
        if "author" in m and m["author"]:
            parts.append("Author: " + m["author"])
        if "category" in m and m["category"]:
            parts.append("Category: " + m["category"])
        if "description" in m and m["description"]:
            parts.append("Description: " + m["description"])
        return " . ".join(parts)

    def _apply_prompt_mask(self, input_ids):
        ids = input_ids.clone()
        mask_token_id = self.tokenizer.mask_token_id

        for i in range(len(ids)):
            if random.random() < self.prompt_mask_ratio:
                ids[i] = mask_token_id
        return ids

    def _apply_prediction_mask(self, target_ids):
        labels = target_ids.clone()
        input_ids = target_ids.clone()

        for i in range(len(input_ids)):
            if random.random() < self.mask_pred_ratio:
                input_ids[i] = self.tokenizer.mask_token_id
            else:
                labels[i] = -100

        return input_ids, labels

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.stage == "id_generation":
            _, item_id = sample

            meta_text = self._build_metadata_text(item_id)

            enc = self.tokenizer(
                meta_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            input_ids = self._apply_prompt_mask(input_ids)

            # For ID initialization, target is the same metadata text projected into ID space
            # In practice, we just learn to reconstruct a pseudo textual ID
            tgt_enc = self.tokenizer(
                meta_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_id_len,
                return_tensors="pt"
            )

            tgt_ids = tgt_enc["input_ids"].squeeze(0)

            dec_input_ids, labels = self._apply_prediction_mask(tgt_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": dec_input_ids,
                "labels": labels
            }

        else:
            _, hist, tgt_item = sample

            hist_texts = []
            for iid in hist:
                hist_texts.append(self._build_metadata_text(iid))

            hist_text = " [HIST] ".join(hist_texts)

            enc = self.tokenizer(
                hist_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            tgt_text = self._build_metadata_text(tgt_item)

            tgt_enc = self.tokenizer(
                tgt_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_id_len,
                return_tensors="pt"
            )

            tgt_ids = tgt_enc["input_ids"].squeeze(0)

            dec_input_ids, labels = self._apply_prediction_mask(tgt_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": dec_input_ids,
                "labels": labels
            }