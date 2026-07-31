"""Core conditional denoising model used by DDGR."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class DiffusionRecommenderModel(nn.Module):
    """Transformer conditional denoiser over continuous sem-id states.

    The same backbone is used with item-side conditions for sem-id grounding and
    with user-side conditions for recommendation. Task-specific behavior comes
    from the condition tokens, not from separate denoising networks.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_identifier_len: int = 32,
        max_condition_len: int = 1024,
        max_diffusion_steps: int = 1000,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_identifier_len = max_identifier_len
        self.max_condition_len = max_condition_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.identifier_position_embedding = nn.Embedding(max_identifier_len, hidden_dim)
        self.condition_position_embedding = nn.Embedding(max_condition_len, hidden_dim)
        self.time_embedding = nn.Embedding(max_diffusion_steps, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.condition_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.denoising_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Legacy aliases retained for older scripts in the repository.
        self.encoder = self.condition_encoder
        self.decoder = self.denoising_decoder
        self.position_embedding = self.identifier_position_embedding

    @staticmethod
    def _positions(length: int, batch_size: int, device: torch.device) -> Tensor:
        return torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)

    def encode_condition(
        self,
        condition_ids: Tensor,
        condition_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if condition_ids.ndim != 2:
            raise ValueError("condition_ids must have shape [B, Lc]")
        batch_size, length = condition_ids.shape
        if length > self.max_condition_len:
            raise ValueError(
                f"condition length {length} exceeds max_condition_len={self.max_condition_len}"
            )
        positions = self._positions(length, batch_size, condition_ids.device)
        condition = self.token_embedding(condition_ids) + self.condition_position_embedding(
            positions
        )
        padding_mask = None
        if condition_attention_mask is not None:
            if condition_attention_mask.shape != condition_ids.shape:
                raise ValueError("condition_attention_mask must match condition_ids")
            padding_mask = ~condition_attention_mask.bool()
        return self.condition_encoder(
            condition, src_key_padding_mask=padding_mask
        )

    def forward_from_embeddings(
        self,
        noisy_embeddings: Tensor,
        condition_ids: Tensor,
        timesteps: Tensor,
        condition_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict token logits from noisy continuous sem-id states."""
        if noisy_embeddings.ndim != 3:
            raise ValueError("noisy_embeddings must have shape [B, K, D]")
        batch_size, length, hidden_dim = noisy_embeddings.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError("noisy embedding dimension does not match hidden_dim")
        if length > self.max_identifier_len:
            raise ValueError(
                f"identifier length {length} exceeds max_identifier_len={self.max_identifier_len}"
            )
        if timesteps.ndim != 1 or timesteps.size(0) != batch_size:
            raise ValueError("timesteps must have shape [B]")

        memory = self.encode_condition(condition_ids, condition_attention_mask)
        positions = self._positions(length, batch_size, noisy_embeddings.device)
        time = self.time_embedding(timesteps).unsqueeze(1)
        decoder_input = (
            noisy_embeddings
            + self.identifier_position_embedding(positions)
            + time
        )
        hidden = self.denoising_decoder(tgt=decoder_input, memory=memory)
        return self.output_projection(hidden)

    def forward(
        self,
        noisy_input_ids: Tensor,
        condition_ids: Tensor,
        timestep: Tensor,
        condition_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compatibility wrapper accepting discrete noisy token ids."""
        noisy_embeddings = self.token_embedding(noisy_input_ids)
        return self.forward_from_embeddings(
            noisy_embeddings,
            condition_ids,
            timestep,
            condition_attention_mask,
        )
