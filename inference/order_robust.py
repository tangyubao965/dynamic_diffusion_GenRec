"""Order-robust candidate scoring for DDGR (Section 3.7)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def order_robust_score(
    token_logits: Tensor,
    candidate_token_ids: Tensor,
    *,
    candidate_mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    normalize_by_length: bool = False,
) -> Tensor:
    """Score candidate sem-ids with the Section 3.7 max aggregation.

    For each token ``x`` in a candidate sem-id, the score selects the most
    compatible predicted output position and then sums across candidate tokens::

        score(u, i) = sum_{x in sem_id(i)} max_k log P(x | u, k).

    Args:
        token_logits: User-conditioned token logits ``[B, K, V]``.
        candidate_token_ids: Candidate identifiers ``[B, C, L]`` or ``[C, L]``.
            A two-dimensional tensor is broadcast to every batch example.
        candidate_mask: Optional boolean mask matching candidate ids.
        ignore_index: Optional token id excluded from scoring, e.g. padding.
        normalize_by_length: Divide the score by the number of valid candidate
            tokens. The paper uses ``False``.

    Returns:
        Candidate scores with shape ``[B, C]``.
    """
    if token_logits.ndim != 3:
        raise ValueError("token_logits must have shape [B, K, V]")
    if candidate_token_ids.ndim == 2:
        candidate_token_ids = candidate_token_ids.unsqueeze(0).expand(
            token_logits.size(0), -1, -1
        )
    if candidate_token_ids.ndim != 3:
        raise ValueError("candidate_token_ids must have shape [C, L] or [B, C, L]")
    if candidate_token_ids.size(0) != token_logits.size(0):
        raise ValueError("candidate and logits batch sizes must match")

    batch_size, positions, vocab_size = token_logits.shape
    _, num_candidates, candidate_len = candidate_token_ids.shape
    candidate_token_ids = candidate_token_ids.long()
    if torch.any((candidate_token_ids < 0) | (candidate_token_ids >= vocab_size)):
        raise ValueError("candidate_token_ids contains values outside the vocabulary")

    valid = torch.ones_like(candidate_token_ids, dtype=torch.bool)
    if candidate_mask is not None:
        if candidate_mask.shape != candidate_token_ids.shape:
            raise ValueError("candidate_mask must match candidate_token_ids")
        valid &= candidate_mask.bool()
    if ignore_index is not None:
        valid &= candidate_token_ids.ne(ignore_index)

    log_probs = F.log_softmax(token_logits, dim=-1)
    # Expand to [B, C, K, V] only virtually, then gather candidate tokens.
    expanded = log_probs.unsqueeze(1).expand(-1, num_candidates, -1, -1)
    gather_ids = candidate_token_ids.unsqueeze(2).expand(
        batch_size, num_candidates, positions, candidate_len
    )
    per_position = torch.gather(expanded, dim=-1, index=gather_ids)
    # [B, C, L]: best output position for every candidate token.
    best_position_log_prob = per_position.max(dim=2).values
    best_position_log_prob = best_position_log_prob.masked_fill(~valid, 0.0)
    scores = best_position_log_prob.sum(dim=-1)

    if normalize_by_length:
        lengths = valid.sum(dim=-1).clamp_min(1).to(scores.dtype)
        scores = scores / lengths
    return scores


def rank_candidates(
    token_logits: Tensor,
    candidate_token_ids: Tensor,
    *,
    candidate_mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Return descending scores and candidate indices."""
    scores = order_robust_score(
        token_logits,
        candidate_token_ids,
        candidate_mask=candidate_mask,
        ignore_index=ignore_index,
    )
    return torch.sort(scores, dim=-1, descending=True)
