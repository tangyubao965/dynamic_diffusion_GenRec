import torch
import numpy as np
from tqdm import tqdm

def evaluate_model(model, dataloader, device, topk=[10, 20]):
    model.eval()
    recall_at_k = {k: [] for k in topk}
    ndcg_at_k = {k: [] for k in topk}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            scores, target_ids = model.predict(batch)

            # scores: [B, num_candidates], target_ids: [B]
            ranks = torch.argsort(scores, dim=1, descending=True)
            for i, k in enumerate(topk):
                topk_preds = ranks[:, :k]
                hits = (topk_preds == target_ids.unsqueeze(1)).any(dim=1).float()
                recall_at_k[k].append(hits.mean().item())

                # NDCG
                true_pos = (topk_preds == target_ids.unsqueeze(1)).float()
                ndcg = true_pos / torch.log2(torch.arange(2, k + 2, device=device).float())
                ndcg_at_k[k].append(ndcg.sum(dim=1).mean().item())

    avg_recall = [np.mean(recall_at_k[k]) for k in topk]
    avg_ndcg = [np.mean(ndcg_at_k[k]) for k in topk]

    return avg_recall[0], avg_recall[1], avg_ndcg[0], avg_ndcg[1]