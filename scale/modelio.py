"""
scale/modelio.py - Self-contained model build + checkpoint I/O for the scale
package. Imports TabularDiffusion directly from diffusion/ (no dependency on the
released src/config.py module-level constants), so the large-scale path is
decoupled from the 27-column release config.
"""
from __future__ import annotations

import os
import torch

import _common  # noqa: F401  (puts diffusion/ on sys.path)
from diffusion import TabularDiffusion


def build_model(input_dim: int, hp: dict) -> TabularDiffusion:
    return TabularDiffusion(
        input_dim=input_dim,
        hidden_dim=hp["hidden_dim"],
        n_layers=hp["n_layers"],
        n_timesteps=hp["n_timesteps"],
        schedule=hp["schedule"],
    )


def save_checkpoint(model: TabularDiffusion, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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