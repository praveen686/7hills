"""Exponential Moving Average of model weights for inference smoothing.

Maintains a shadow copy of model parameters updated via:
    ema_param = decay * ema_param + (1 - decay) * model_param

This produces smoother, more robust predictions by averaging over the
optimization trajectory. Standard practice since Polyak & Juditsky (1992),
widely used in modern deep learning (SWA, EMA in diffusion models, etc.).

Usage during training::

    ema = ModelEMA(model, decay=0.999)
    for batch in dataloader:
        loss.backward()
        optimizer.step()
        ema.update()  # update EMA after each optimizer step

    # For inference:
    ema.apply()       # copy EMA weights to model
    model.eval()
    predictions = model(x)
    ema.restore()     # restore original weights

    # Or use the context manager:
    with ema.average_parameters():
        model.eval()
        predictions = model(x)
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available; ModelEMA will not function")


class ModelEMA:
    """Exponential Moving Average of model weights for inference smoothing.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to track.
    decay : float
        EMA decay factor in [0, 1). Higher values give smoother averaging.
        Typical values: 0.999 (slow, smooth), 0.99 (faster adaptation).

    Attributes
    ----------
    shadow : dict[str, torch.Tensor]
        Shadow (EMA) copies of model parameters.
    backup : dict[str, torch.Tensor]
        Backup of original weights during apply/restore cycle.
    """

    def __init__(self, model: "nn.Module", decay: float = 0.999):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for ModelEMA")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self.model = model
        self.decay = decay
        self.shadow: dict[str, "torch.Tensor"] = {}
        self.backup: dict[str, "torch.Tensor"] = {}
        self._init_shadow()

    def _init_shadow(self) -> None:
        """Initialize shadow params as a copy of model params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow params with EMA step.

        shadow[name] = decay * shadow[name] + (1 - decay) * param.data
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """Copy EMA weights to model (for inference). Call restore() after."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after inference."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Context manager that applies EMA weights and restores on exit.

        Usage::

            with ema.average_parameters():
                model.eval()
                output = model(x)
        """
        self.apply()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpoint saving.

        Returns
        -------
        dict
            Contains 'shadow' (parameter tensors moved to CPU) and 'decay'.
        """
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EMA state from checkpoint.

        Parameters
        ----------
        state : dict
            Output of ``state_dict()``.
        """
        self.decay = state["decay"]
        for k, v in state["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))

    def __repr__(self) -> str:
        n_params = len(self.shadow)
        total_elements = sum(v.numel() for v in self.shadow.values())
        return (
            f"ModelEMA(decay={self.decay}, "
            f"tracked_params={n_params}, "
            f"total_elements={total_elements:,})"
        )
