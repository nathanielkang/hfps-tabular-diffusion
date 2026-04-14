"""
model.py - Thin wrapper around TabularDiffusion with checkpoint I/O.
"""

import os
import sys
import torch

from config import DIFFUSION_DIR

if DIFFUSION_DIR not in sys.path:
    sys.path.insert(0, DIFFUSION_DIR)

from diffusion import TabularDiffusion  # noqa: E402


def build_model(input_dim: int, hp: dict) -> TabularDiffusion:
    """Construct a TabularDiffusion model from a hyperparameter dict."""
    return TabularDiffusion(
        input_dim=input_dim,
        hidden_dim=hp["hidden_dim"],
        n_layers=hp["n_layers"],
        n_timesteps=hp["n_timesteps"],
        schedule=hp["schedule"],
    )


def save_checkpoint(model: TabularDiffusion, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: TabularDiffusion, path: str) -> TabularDiffusion:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    if any(k.startswith("ema_denoiser.") for k in state):
        model._init_ema()
        ema_state = {k.replace("ema_denoiser.", ""): v
                     for k, v in state.items() if k.startswith("ema_denoiser.")}
        model.ema_denoiser.load_state_dict(ema_state)
    return model
