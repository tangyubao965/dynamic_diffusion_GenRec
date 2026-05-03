import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionRecommenderModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, dropout=0.1, max_len=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.time_embedding = nn.Embedding(1000, hidden_dim)

    def add_positional_encoding(self, x):
        B, L = x.size()
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.token_embedding(x) + self.position_embedding(positions)

    def forward_diffusion_step(self, noisy_input, t_embed, condition):
        B, L, D = noisy_input.size()
        condition_encoded = self.encoder(condition)
        t_embed = t_embed.unsqueeze(1).expand(B, L, D)
        decoder_input = noisy_input + t_embed
        out = self.decoder(tgt=decoder_input, memory=condition_encoded)
        return self.output_projection(out)

    def get_timestep_embedding(self, t, device):
        return self.time_embedding(t)

    def forward(self, noisy_input_ids, condition_ids, timestep):
        x_t = self.token_embedding(noisy_input_ids)
        cond = self.add_positional_encoding(condition_ids)
        t_embed = self.get_timestep_embedding(timestep, device=x_t.device)
        logits = self.forward_diffusion_step(x_t, t_embed, cond)
        return logits