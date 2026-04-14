"""
diffusion.py - Simplified TabDDPM (Denoising Diffusion Probabilistic Model)
for tabular data.

Implements:
    - Forward (noising) process with linear or cosine beta schedule
    - Sinusoidal timestep embeddings
    - MLP-based denoising network  eps_theta(x_t, t)  with residual connections
    - Standard DDPM training loss  L = E[||eps - eps_theta(x_t, t)||^2]
    - Reverse-process sampling with value clamping to prevent divergence
    - EMA (Exponential Moving Average) of weights for stable sampling

Designed to run on CPU (32 GB RAM).
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(n_timesteps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999).float()


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Residual MLP block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DenoisingMLP(nn.Module):
    def __init__(self, input_dim: int, time_dim: int,
                 hidden_dim: int = 256, n_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([x_t, t_emb], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# TabularDiffusion (unconditional DDPM) with EMA
# ---------------------------------------------------------------------------

class TabularDiffusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 3, n_timesteps: int = 1000,
                 schedule: str = "linear"):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps

        if schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",
                             torch.sqrt(1.0 - alpha_bar))

        time_dim = min(128, hidden_dim)
        self.denoiser = DenoisingMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=0.0,
        )

        self.ema_denoiser = None

    def _init_ema(self):
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.ema_denoiser.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self, decay: float = 0.999):
        if self.ema_denoiser is None:
            return
        for ema_p, p in zip(self.ema_denoiser.parameters(),
                            self.denoiser.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x, t)

    def compute_loss(self, x_0: torch.Tensor,
                     sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = self.forward(x_t, t)
        loss_per_sample = (noise - predicted_noise).pow(2).mean(dim=-1)
        if sample_weights is not None:
            loss = (loss_per_sample * sample_weights).mean()
        else:
            loss = loss_per_sample.mean()
        return loss

    def train_model(self, X_train: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 1e-3,
                    verbose: bool = True) -> list:
        self.train()
        self._init_ema()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        losses = []
        epoch_iter = tqdm(range(epochs), desc="TabDDPM training",
                          disable=not verbose)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            n_batches = 0
            for (x_batch,) in loader:
                optimizer.zero_grad()
                loss = self.compute_loss(x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                self._update_ema()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        return losses

    @torch.no_grad()
    def sample(self, n_samples: int, verbose: bool = False) -> np.ndarray:
        self.eval()
        net = self.ema_denoiser if self.ema_denoiser is not None else self.denoiser

        x = torch.randn(n_samples, self.input_dim)

        timesteps = list(range(self.n_timesteps - 1, -1, -1))
        step_iter = tqdm(timesteps, desc="Sampling", disable=not verbose)

        for t_val in step_iter:
            t = torch.full((n_samples,), t_val, dtype=torch.long)
            t_emb = net.time_embed(t)
            inp = torch.cat([x, t_emb], dim=-1)
            h = net.input_proj(inp)
            for block in net.blocks:
                h = block(h)
            predicted_noise = net.output_proj(h)

            alpha = self.alphas[t_val]
            alpha_b = self.alpha_bar[t_val]
            beta = self.betas[t_val]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_b)
            mean = coef1 * (x - coef2 * predicted_noise)

            if t_val > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

            x = torch.clamp(x, -6.0, 6.0)

        return x.numpy()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(500, 8).astype(np.float32)

    model = TabularDiffusion(input_dim=8, hidden_dim=128, n_layers=2,
                             n_timesteps=100)
    print(model)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = model.train_model(X_dummy, epochs=5, batch_size=64, lr=1e-3)
    print(f"Final loss: {losses[-1]:.4f}")

    samples = model.sample(10)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(samples[:3])
