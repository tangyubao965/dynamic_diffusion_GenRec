"""Loss functions used by DDGR.

This module contains the set-aware objective from Eq. (5) of the paper.
The implementation is written in log space for numerical stability.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def set_aware_loss(
    logits: Tensor,
    target_token_ids: Tensor,
    *,
    target_mask: Optional[Tensor] = None,
    position_mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute the set-aware token objective in Eq. (5).

    For every predicted sem-id position ``k``, the loss marginalizes over all
    valid tokens in the target sem-id rather than fixing a token-to-position
    assignment::

        -log(1 / |S_i| * sum_{x in S_i} p(x | position k)).

    Args:
        logits: Token logits with shape ``[batch, positions, vocabulary]``.
        target_token_ids: Target sem-id tokens with shape
            ``[batch, target_tokens]``. The target order is not used.
        target_mask: Optional boolean mask with the same shape as
            ``target_token_ids``. ``True`` marks valid target tokens.
        position_mask: Optional boolean mask with shape ``[batch, positions]``.
            ``True`` marks output positions that contribute to the loss.
        ignore_index: Optional token id to exclude from the target set.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``.

    Returns:
        A scalar loss for ``sum``/``mean``, or a ``[batch, positions]`` tensor
        for ``none``.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, K, V], got {tuple(logits.shape)}")
    if target_token_ids.ndim != 2:
        raise ValueError(
            "target_token_ids must have shape [B, T], "
            f"got {tuple(target_token_ids.shape)}"
        )
    if logits.size(0) != target_token_ids.size(0):
        raise ValueError("logits and target_token_ids must have the same batch size")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    batch_size, positions, vocab_size = logits.shape
    target_token_ids = target_token_ids.long()
    if torch.any((target_token_ids < 0) | (target_token_ids >= vocab_size)):
        raise ValueError("target_token_ids contains values outside the vocabulary")

    valid_targets = torch.ones_like(target_token_ids, dtype=torch.bool)
    if target_mask is not None:
        if target_mask.shape != target_token_ids.shape:
            raise ValueError("target_mask must match target_token_ids")
        valid_targets &= target_mask.bool()
    if ignore_index is not None:
        valid_targets &= target_token_ids.ne(ignore_index)

    target_count = valid_targets.sum(dim=-1)
    if torch.any(target_count == 0):
        raise ValueError("each example must contain at least one valid target token")

    log_probs = F.log_softmax(logits, dim=-1)
    gather_ids = target_token_ids.unsqueeze(1).expand(batch_size, positions, -1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=gather_ids)
    target_log_probs = target_log_probs.masked_fill(
        ~valid_targets.unsqueeze(1), float("-inf")
    )

    # log(mean_j p(target_j | position_k))
    log_mean_prob = torch.logsumexp(target_log_probs, dim=-1)
    log_mean_prob = log_mean_prob - target_count.to(logits.dtype).log().unsqueeze(1)
    loss = -log_mean_prob

    if position_mask is not None:
        if position_mask.shape != loss.shape:
            raise ValueError("position_mask must have shape [B, K]")
        valid_positions = position_mask.bool()
    else:
        valid_positions = torch.ones_like(loss, dtype=torch.bool)

    if reduction == "none":
        return loss.masked_fill(~valid_positions, 0.0)

    selected = loss[valid_positions]
    if selected.numel() == 0:
        raise ValueError("position_mask selects no output positions")
    if reduction == "sum":
        return selected.sum()
    return selected.mean()


def diffusion_loss(predicted_noise: Tensor, target_noise: Tensor) -> Tensor:
    """Standard DDPM noise-prediction loss."""
    return F.mse_loss(predicted_noise, target_noise)


def mask_prediction_loss(
    logits: Tensor,
    target_ids: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Position-wise token cross entropy used by legacy training scripts."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, K, V]")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )
