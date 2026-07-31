"""Continuous diffusion utilities for DDGR."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.losses import set_aware_loss


class DiffusionScheduler:
    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas = torch.linspace(
            beta_start, beta_end, num_steps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device: str | torch.device) -> "DiffusionScheduler":
        self.device = torch.device(device)
        for name in (
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    def sample_timesteps(self, batch_size: int) -> Tensor:
        return torch.randint(
            0,
            self.num_steps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def q_sample(
        self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

    def inference_timesteps(self, num_inference_steps: int) -> Tensor:
        if num_inference_steps < 1 or num_inference_steps > self.num_steps:
            raise ValueError("num_inference_steps must be in [1, num_steps]")
        return torch.linspace(
            self.num_steps - 1,
            0,
            steps=num_inference_steps,
            device=self.device,
        ).round().long().unique_consecutive()


class DiffusionTrainer:
    def __init__(self, model, scheduler: DiffusionScheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    def compute_losses(
        self,
        clean_input_ids: Tensor,
        condition_ids: Tensor,
        *,
        target_set_ids: Optional[Tensor] = None,
        target_set_mask: Optional[Tensor] = None,
        condition_attention_mask: Optional[Tensor] = None,
        lambda_set: float = 1.0,
        ignore_index: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Compute diffusion recovery and Eq. (5) set-aware losses."""
        batch_size = clean_input_ids.size(0)
        t = self.scheduler.sample_timesteps(batch_size)
        x0 = self.model.token_embedding(clean_input_ids)
        x_t, noise = self.scheduler.q_sample(x0, t)

        logits = self.model.forward_from_embeddings(
            noisy_embeddings=x_t,
            condition_ids=condition_ids,
            timesteps=t,
            condition_attention_mask=condition_attention_mask,
        )
        probs = F.softmax(logits, dim=-1)
        pred_x0 = torch.matmul(probs, self.model.token_embedding.weight)
        sqrt_ab = self.scheduler.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.scheduler.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        pred_noise = (x_t - sqrt_ab * pred_x0) / (sqrt_1mab + 1e-8)
        diffusion = F.mse_loss(pred_noise, noise)

        if target_set_ids is None:
            target_set_ids = clean_input_ids
        set_loss = set_aware_loss(
            logits,
            target_set_ids,
            target_mask=target_set_mask,
            ignore_index=ignore_index,
        )
        total = diffusion + float(lambda_set) * set_loss
        return {
            "total_loss": total,
            "diffusion_loss": diffusion,
            "set_loss": set_loss,
            "logits": logits,
        }

    def compute_loss(self, clean_input_ids: Tensor, condition_ids: Tensor, **kwargs) -> Tensor:
        """Backward-compatible scalar loss wrapper."""
        return self.compute_losses(clean_input_ids, condition_ids, **kwargs)[
            "total_loss"
        ]

    @torch.no_grad()
    def sample_with_logits(
        self,
        condition_ids: Tensor,
        seq_len: int,
        *,
        num_inference_steps: int = 10,
        condition_attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Fast deterministic reverse diffusion returning tokens and logits."""
        device = condition_ids.device
        self.scheduler.to(device)
        batch_size = condition_ids.size(0)
        x_t = torch.randn(
            batch_size, seq_len, self.model.hidden_dim, device=device
        )
        timesteps = self.scheduler.inference_timesteps(num_inference_steps)
        final_logits = None

        for index, t_value in enumerate(timesteps):
            t = int(t_value.item())
            t_batch = torch.full(
                (batch_size,), t, device=device, dtype=torch.long
            )
            logits = self.model.forward_from_embeddings(
                noisy_embeddings=x_t,
                condition_ids=condition_ids,
                timesteps=t_batch,
                condition_attention_mask=condition_attention_mask,
            )
            probs = F.softmax(logits, dim=-1)
            pred_x0 = torch.matmul(probs, self.model.token_embedding.weight)
            final_logits = logits

            if index == len(timesteps) - 1:
                x_t = pred_x0
                break

            next_t = int(timesteps[index + 1].item())
            alpha_bar_t = self.scheduler.alpha_bars[t]
            alpha_bar_next = self.scheduler.alpha_bars[next_t]
            eps = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / (
                torch.sqrt(1.0 - alpha_bar_t) + 1e-8
            )
            x_t = (
                torch.sqrt(alpha_bar_next) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_next) * eps
            )

        if final_logits is None:
            raise RuntimeError("reverse diffusion produced no logits")
        # Re-project the final continuous state for a consistent final decode.
        final_logits = self.model.output_projection(x_t)
        tokens = final_logits.argmax(dim=-1)
        return tokens, final_logits

    @torch.no_grad()
    def sample(
        self,
        condition_ids: Tensor,
        seq_len: int,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        tokens, _ = self.sample_with_logits(
            condition_ids,
            seq_len,
            num_inference_steps=num_steps or 10,
            **kwargs,
        )
        return tokens
