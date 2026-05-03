import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionScheduler:
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long
        )

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        x0: (B, L, D)
        t: (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - sqrt_1mab * eps) / (sqrt_ab + 1e-8)


class DiffusionTrainer:
    def __init__(self, model, scheduler: DiffusionScheduler):
        self.model = model
        self.scheduler = scheduler

    def compute_loss(self, clean_input_ids, condition_ids):
        """
        clean_input_ids: (B, L)  target identifier tokens
        condition_ids:   (B, Lc) metadata tokens or history tokens
        """

        device = clean_input_ids.device
        B, L = clean_input_ids.size()

        t = self.scheduler.sample_timesteps(B)

        # Embed clean tokens
        x0 = self.model.token_embedding(clean_input_ids)  # (B, L, D)

        # Forward diffusion
        x_t, noise = self.scheduler.q_sample(x0, t)

        # Predict noise with model
        logits = self.model.forward_from_embeddings(
            noisy_embeddings=x_t,
            condition_ids=condition_ids,
            timesteps=t
        )

        # Project logits to embedding space via softmax * embedding matrix
        # We use predicted token distribution to reconstruct embedding
        probs = F.softmax(logits, dim=-1)  # (B, L, V)
        emb_table = self.model.token_embedding.weight  # (V, D)
        pred_x0 = torch.matmul(probs, emb_table)  # (B, L, D)

        # Predict noise from pred_x0
        sqrt_ab = self.scheduler.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.scheduler.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        pred_noise = (x_t - sqrt_ab * pred_x0) / (sqrt_1mab + 1e-8)

        loss = F.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def sample(
        self,
        condition_ids,
        seq_len: int,
        num_steps: int = None
    ):
        """
        DDPM sampling loop
        condition_ids: (B, Lc)
        returns: token ids (B, L)
        """
        device = condition_ids.device
        if num_steps is None:
            num_steps = self.scheduler.num_steps

        B = condition_ids.size(0)
        D = self.model.hidden_dim

        x_t = torch.randn(B, seq_len, D, device=device)

        for t in reversed(range(num_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            logits = self.model.forward_from_embeddings(
                noisy_embeddings=x_t,
                condition_ids=condition_ids,
                timesteps=t_batch
            )

            probs = F.softmax(logits, dim=-1)
            emb_table = self.model.token_embedding.weight
            pred_x0 = torch.matmul(probs, emb_table)

            beta = self.scheduler.betas[t]
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = coef1 * (x_t - coef2 * (x_t - pred_x0)) + torch.sqrt(beta) * noise

        # Final decode
        logits = self.model.output_projection(x_t)
        tokens = torch.argmax(logits, dim=-1)
        return tokens