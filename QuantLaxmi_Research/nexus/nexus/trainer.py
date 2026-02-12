"""NEXUS Training Pipeline -- 3-phase training for the world model.

Phase 1: JEPA Pre-training (Self-Supervised)
    Learn market representations by predicting masked future states in latent space.
    No reward signal needed -- pure representation learning.
    Loss: ||z_hat_target - z_target||^2 + lambda_hyp * d_H(pred, target)

Phase 2: World Model Training (TD Learning)
    Learn dynamics, reward, and value models using replay buffer.
    Transitions: (z, a, r, z_next, done)
    Loss: dynamics_loss + reward_loss + value_loss (from planner.compute_td_loss)

Phase 3: Policy Distillation
    Distill the CEM planner into the direct policy head for fast inference.
    teacher = planner.plan(z)  ->  student = policy_head(z)
    Loss: MSE(policy_head(z), planner_action)

Usage
-----
    from nexus.model import create_nexus
    from nexus.trainer import NexusTrainer, TrainerConfig

    model = create_nexus(n_features=192, n_assets=6, size="base")
    trainer = NexusTrainer(model, TrainerConfig())
    trainer.train(train_loader, val_loader)

References
----------
    V-JEPA 2: Meta AI, June 2025 (EMA schedule, masking strategy)
    TD-MPC2: Hansen et al., ICLR 2024 (world model + CEM planning)
    DreamerV3: Hafner et al., Nature 2025 (3-phase training)
"""

from __future__ import annotations

import gc
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

# Conditional imports for mixed precision
try:
    from torch.amp import autocast, GradScaler  # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch < 2.4

from .model import NEXUS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Configuration for the 3-phase NEXUS training pipeline.

    Parameters are grouped by training phase with sensible defaults
    tuned for T4 GPU (16 GB VRAM).
    """

    # -- Phase 1: JEPA Pre-training ----------------------------------------
    jepa_epochs: int = 50
    jepa_lr: float = 1e-4
    jepa_weight_decay: float = 1e-5
    jepa_warmup_steps: int = 1000
    mask_ratio: float = 0.4
    ema_schedule: Tuple[float, float] = (0.996, 0.9999)
    hyp_weight: float = 0.1

    # -- Phase 2: World Model Training -------------------------------------
    world_epochs: int = 30
    world_lr: float = 3e-4
    world_weight_decay: float = 1e-5
    world_batch_size: int = 256
    discount: float = 0.99
    replay_capacity: int = 100_000

    # -- Phase 3: Policy Distillation --------------------------------------
    distill_epochs: int = 20
    distill_lr: float = 1e-4
    distill_weight_decay: float = 1e-5
    n_distill_samples: int = 10_000
    distill_batch_size: int = 256

    # -- General -----------------------------------------------------------
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp: bool = True
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2000

    def __post_init__(self) -> None:
        """Disable AMP when CUDA is not available."""
        if self.device == "cpu":
            self.use_amp = False


# ---------------------------------------------------------------------------
# Cosine Warmup Scheduler
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    """Learning rate scheduler: linear warmup followed by cosine decay.

    During warmup (step < warmup_steps):
        lr = base_lr * step / warmup_steps

    After warmup:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    where progress = (step - warmup_steps) / (total_steps - warmup_steps).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be adjusted.
    base_lr : float
        Peak learning rate (reached at end of warmup).
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps (warmup + cosine decay).
    min_lr : float
        Minimum learning rate at end of cosine schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.min_lr = min_lr
        self._step = 0

    def step(self) -> float:
        """Advance one step and update the learning rate.

        Returns
        -------
        float
            The new learning rate.
        """
        self._step += 1
        lr = self.get_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self, step: int) -> float:
        """Compute learning rate for a given step.

        Parameters
        ----------
        step : int
            Current training step (1-indexed).

        Returns
        -------
        float
            The learning rate at *step*.
        """
        if step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / max(self.warmup_steps, 1)

        # Cosine decay
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer storing (z, a, r, z_next, done) transitions for
    world model training.

    All tensors are stored detached on CPU to avoid GPU memory pressure.
    Batches are moved to the target device on :meth:`sample`.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored (FIFO eviction).
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    # -- Mutation -----------------------------------------------------------

    def add(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        z_next: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store a batch of transitions.

        Parameters
        ----------
        z : (B, d_latent) -- current latent state
        a : (B, d_action) -- action taken
        r : (B, 1) -- observed reward
        z_next : (B, d_latent) -- next latent state
        done : (B, 1) -- terminal flag (1.0 = terminal)
        """
        B = z.size(0)
        for i in range(B):
            self.buffer.append((
                z[i].detach().cpu(),
                a[i].detach().cpu(),
                r[i].detach().cpu(),
                z_next[i].detach().cpu(),
                done[i].detach().cpu(),
            ))

    # -- Sampling -----------------------------------------------------------

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : str
            Device to place the returned tensors on.

        Returns
        -------
        dict
            Keys: ``z``, ``a``, ``r``, ``z_next``, ``done`` -- each a tensor
            on *device*.
        """
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        z_list, a_list, r_list, zn_list, d_list = [], [], [], [], []
        for idx in indices:
            z_i, a_i, r_i, zn_i, d_i = self.buffer[idx]
            z_list.append(z_i)
            a_list.append(a_i)
            r_list.append(r_i)
            zn_list.append(zn_i)
            d_list.append(d_i)
        return {
            "z": torch.stack(z_list).to(device),
            "a": torch.stack(a_list).to(device),
            "r": torch.stack(r_list).to(device),
            "z_next": torch.stack(zn_list).to(device),
            "done": torch.stack(d_list).to(device),
        }

    def sample_iterator(
        self, batch_size: int, device: str = "cpu"
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield batches that cover the entire buffer once (one epoch).

        Parameters
        ----------
        batch_size : int
            Size of each mini-batch.
        device : str
            Device for output tensors.

        Yields
        ------
        dict
            Same structure as :meth:`sample`.
        """
        n = len(self.buffer)
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            indices = perm[start:end]
            z_list, a_list, r_list, zn_list, d_list = [], [], [], [], []
            for idx in indices:
                z_i, a_i, r_i, zn_i, d_i = self.buffer[idx]
                z_list.append(z_i)
                a_list.append(a_i)
                r_list.append(r_i)
                zn_list.append(zn_i)
                d_list.append(d_i)
            yield {
                "z": torch.stack(z_list).to(device),
                "a": torch.stack(a_list).to(device),
                "r": torch.stack(r_list).to(device),
                "z_next": torch.stack(zn_list).to(device),
                "done": torch.stack(d_list).to(device),
            }

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# NEXUS Trainer
# ---------------------------------------------------------------------------

class NexusTrainer:
    """Complete 3-phase training pipeline for NEXUS.

    Phase 1 -- JEPA Pre-training (self-supervised representation learning)
    Phase 2 -- World Model Training (TD learning on replay buffer)
    Phase 3 -- Policy Distillation (CEM planner -> direct policy head)

    Parameters
    ----------
    model : NEXUS
        The NEXUS model instance to train.
    config : TrainerConfig
        Training hyperparameters.
    """

    def __init__(self, model: NEXUS, config: Optional[TrainerConfig] = None):
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # -- Phase 1 optimiser: JEPA encoder + predictor -------------------
        jepa_params = (
            list(model.world_model.context_encoder.parameters())
            + list(model.world_model.context_proj.parameters())
            + list(model.world_model.predictor.parameters())
        )
        if model.world_model.use_hyperbolic:
            jepa_params += list(model.world_model.to_hyperbolic.parameters())
            jepa_params += list(model.world_model.from_hyperbolic.parameters())
        self.jepa_optimizer = AdamW(
            jepa_params,
            lr=self.config.jepa_lr,
            weight_decay=self.config.jepa_weight_decay,
        )

        # -- Phase 2 optimiser: planner (dynamics + reward + value) --------
        world_params = list(model.planner.parameters()) + list(
            model.world_model.reward_head.parameters()
        ) + list(model.world_model.value_head.parameters())
        self.world_optimizer = AdamW(
            world_params,
            lr=self.config.world_lr,
            weight_decay=self.config.world_weight_decay,
        )

        # -- Phase 3 optimiser: policy head only ---------------------------
        self.policy_optimizer = AdamW(
            model.policy_head.parameters(),
            lr=self.config.distill_lr,
            weight_decay=self.config.distill_weight_decay,
        )

        # -- Mixed precision -----------------------------------------------
        self.scaler = GradScaler(enabled=self.config.use_amp)

        # -- Replay buffer --------------------------------------------------
        self.replay_buffer = ReplayBuffer(capacity=self.config.replay_capacity)

        # -- Bookkeeping ----------------------------------------------------
        self.global_step = 0
        self.metrics_history: List[Dict[str, float]] = []

    # ======================================================================
    #  Phase 1 -- JEPA Pre-training
    # ======================================================================

    def train_jepa(
        self,
        dataloader: Any,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Self-supervised JEPA pre-training on market data.

        Learns market representations by predicting masked future latent
        states. The target encoder is updated with an exponential moving
        average (EMA) of the context encoder weights, with the decay
        linearly warmed from ``ema_schedule[0]`` to ``ema_schedule[1]``.

        Parameters
        ----------
        dataloader : iterable
            Yields ``(context, target, target_positions)`` batches.
            * context : (B, L_ctx, n_features)
            * target  : (B, L_tgt, n_features)
            * target_positions : (B, L_tgt) int
        epochs : int, optional
            Override ``config.jepa_epochs``.

        Returns
        -------
        dict
            Lists of per-step losses keyed by ``"jepa_loss"``,
            ``"hyperbolic_loss"``, and ``"total_loss"``.
        """
        epochs = epochs or self.config.jepa_epochs
        cfg = self.config
        self.model.train()

        # Estimate total steps for scheduler
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = 100  # fallback for infinite iterators
        total_steps = epochs * steps_per_epoch

        scheduler = CosineWarmupScheduler(
            self.jepa_optimizer,
            base_lr=cfg.jepa_lr,
            warmup_steps=cfg.jepa_warmup_steps,
            total_steps=total_steps,
        )

        history: Dict[str, List[float]] = {
            "jepa_loss": [],
            "hyperbolic_loss": [],
            "total_loss": [],
        }
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                context, target, target_pos = self._unpack_jepa_batch(batch)

                # EMA decay warmup: linear from ema_schedule[0] -> ema_schedule[1]
                ema_decay = self._get_ema_decay(step, total_steps)
                self.model.world_model.ema_decay = ema_decay

                # Forward with AMP
                amp_device = "cuda" if self.device.type == "cuda" else "cpu"
                with autocast(amp_device, enabled=cfg.use_amp):
                    out = self.model(context, target, target_pos)
                    loss = (
                        out["jepa_loss"]
                        + cfg.hyp_weight * out["hyperbolic_loss"]
                    )

                # Backward
                self.jepa_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.jepa_optimizer)
                clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.jepa_optimizer)
                self.scaler.update()

                # EMA update of target encoder
                self.model.world_model._update_target_encoder()

                # LR schedule
                scheduler.step()

                # Record
                loss_val = loss.item()
                history["jepa_loss"].append(out["jepa_loss"].item())
                history["hyperbolic_loss"].append(out["hyperbolic_loss"].item())
                history["total_loss"].append(loss_val)
                epoch_loss += loss_val
                n_batches += 1
                step += 1
                self.global_step += 1

                # Log
                if step % cfg.log_every == 0:
                    lr = scheduler.get_lr(step)
                    self._log(
                        phase="JEPA",
                        step=step,
                        total=total_steps,
                        loss=loss_val,
                        extras={
                            "jepa": out["jepa_loss"].item(),
                            "hyp": out["hyperbolic_loss"].item(),
                            "ema": ema_decay,
                            "lr": lr,
                        },
                    )

                # Checkpoint
                if step % cfg.save_every == 0:
                    self._save_checkpoint(f"jepa_step{step}")

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  [JEPA] Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.6f}")

        return history

    # ======================================================================
    #  Phase 2 -- World Model Training
    # ======================================================================

    def train_world_model(
        self,
        dataloader: Any,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train dynamics, reward, and value models via TD learning.

        First collects transitions into the replay buffer by running the
        current policy on the provided data, then trains from replay.

        Parameters
        ----------
        dataloader : iterable
            Yields ``(context, target, target_positions)`` or dict batches
            with optional ``"rewards"`` key.
        epochs : int, optional
            Override ``config.world_epochs``.

        Returns
        -------
        dict
            Lists of per-step losses: ``"dynamics_loss"``,
            ``"reward_loss"``, ``"value_loss"``, ``"total_loss"``.
        """
        epochs = epochs or self.config.world_epochs
        cfg = self.config
        self.model.train()

        # Step 1: Collect transitions into replay buffer
        print("  [World] Collecting transitions...")
        self._collect_transitions(dataloader)
        print(f"  [World] Replay buffer size: {len(self.replay_buffer):,}")

        if len(self.replay_buffer) < cfg.world_batch_size:
            print("  [World] WARNING: Not enough transitions for a full batch. "
                  f"Have {len(self.replay_buffer)}, need {cfg.world_batch_size}.")
            # Use whatever we have
            batch_size = max(1, len(self.replay_buffer))
        else:
            batch_size = cfg.world_batch_size

        history: Dict[str, List[float]] = {
            "dynamics_loss": [],
            "reward_loss": [],
            "value_loss": [],
            "total_loss": [],
        }
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.replay_buffer.sample_iterator(
                batch_size, device=str(self.device)
            ):
                z = batch["z"]
                a = batch["a"]
                r = batch["r"]
                z_next = batch["z_next"]
                done = batch["done"]

                amp_device = "cuda" if self.device.type == "cuda" else "cpu"
                with autocast(amp_device, enabled=cfg.use_amp):
                    losses = self.model.planner.compute_td_loss(
                        z, a, r, z_next, done, discount=cfg.discount,
                    )

                self.world_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.unscale_(self.world_optimizer)
                clip_grad_norm_(self.model.planner.parameters(), cfg.grad_clip)
                self.scaler.step(self.world_optimizer)
                self.scaler.update()

                # Record
                loss_val = losses["total_loss"].item()
                history["dynamics_loss"].append(losses["dynamics_loss"].item())
                history["reward_loss"].append(losses["reward_loss"].item())
                history["value_loss"].append(losses["value_loss"].item())
                history["total_loss"].append(loss_val)
                epoch_loss += loss_val
                n_batches += 1
                step += 1
                self.global_step += 1

                # Log
                if step % cfg.log_every == 0:
                    self._log(
                        phase="World",
                        step=step,
                        total=epochs * max(
                            len(self.replay_buffer) // batch_size, 1
                        ),
                        loss=loss_val,
                        extras={
                            "dyn": losses["dynamics_loss"].item(),
                            "rew": losses["reward_loss"].item(),
                            "val": losses["value_loss"].item(),
                        },
                    )

                # Checkpoint
                if step % cfg.save_every == 0:
                    self._save_checkpoint(f"world_step{step}")

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  [World] Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.6f}")

        return history

    # ======================================================================
    #  Phase 3 -- Policy Distillation
    # ======================================================================

    def distill_policy(
        self,
        dataloader: Any,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Distill the CEM planner into the direct policy head.

        Freezes the entire model except ``policy_head``, then trains the
        policy to imitate CEM-planned actions.

        Parameters
        ----------
        dataloader : iterable
            Yields batches containing at least a ``context`` tensor
            (B, L, n_features).
        epochs : int, optional
            Override ``config.distill_epochs``.

        Returns
        -------
        dict
            List of per-step ``"distill_loss"`` values.
        """
        epochs = epochs or self.config.distill_epochs
        cfg = self.config

        # Freeze everything except policy_head
        frozen_params: List[nn.Parameter] = []
        for p in self.model.parameters():
            if p.requires_grad:
                frozen_params.append(p)
                p.requires_grad = False
        for p in self.model.policy_head.parameters():
            p.requires_grad = True

        self.model.eval()  # BN / dropout in eval mode for teacher
        # But policy_head trains
        self.model.policy_head.train()

        history: Dict[str, List[float]] = {"distill_loss": []}
        step = 0

        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = 100

        total_steps = epochs * steps_per_epoch

        scheduler = CosineWarmupScheduler(
            self.policy_optimizer,
            base_lr=cfg.distill_lr,
            warmup_steps=min(200, total_steps // 5),
            total_steps=total_steps,
        )

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                n_batches = 0

                for batch in dataloader:
                    context = self._extract_context(batch)

                    # Encode to latent space (no grad -- frozen encoder)
                    with torch.no_grad():
                        z = self.model.encode(context)

                        # Teacher: CEM planner
                        teacher_action, _ = self.model.planner.plan(z)

                    # Student: direct policy head (with grad)
                    amp_device = "cuda" if self.device.type == "cuda" else "cpu"
                    with autocast(amp_device, enabled=cfg.use_amp):
                        student_action = (
                            self.model.policy_head(z) * self.model.cfg.max_position
                        )
                        loss = F.mse_loss(student_action, teacher_action)

                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.policy_optimizer)
                    clip_grad_norm_(self.model.policy_head.parameters(), cfg.grad_clip)
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.update()

                    scheduler.step()

                    loss_val = loss.item()
                    history["distill_loss"].append(loss_val)
                    epoch_loss += loss_val
                    n_batches += 1
                    step += 1
                    self.global_step += 1

                    if step % cfg.log_every == 0:
                        self._log(
                            phase="Distill",
                            step=step,
                            total=total_steps,
                            loss=loss_val,
                            extras={"lr": scheduler.get_lr(step)},
                        )

                    if step % cfg.save_every == 0:
                        self._save_checkpoint(f"distill_step{step}")

                avg_loss = epoch_loss / max(n_batches, 1)
                print(
                    f"  [Distill] Epoch {epoch + 1}/{epochs}  "
                    f"avg_loss={avg_loss:.6f}"
                )

        finally:
            # Unfreeze all previously frozen parameters
            for p in frozen_params:
                p.requires_grad = True
            self.model.train()

        return history

    # ======================================================================
    #  Full Pipeline
    # ======================================================================

    def train(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the complete 3-phase training pipeline.

        Parameters
        ----------
        train_loader : iterable
            Training data loader (used in all 3 phases).
        val_loader : iterable, optional
            Validation data loader for evaluation between phases.

        Returns
        -------
        dict
            Combined metrics from all 3 phases.
        """
        print("=" * 60)
        print("NEXUS Training Pipeline")
        print(f"  Device   : {self.device}")
        print(f"  AMP      : {self.config.use_amp}")
        print(f"  Params   : {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

        results: Dict[str, Any] = {}

        # Phase 1: JEPA Pre-training
        self._clear_gpu_memory()
        print("\n[Phase 1/3] JEPA Pre-training...")
        t0 = time.time()
        results["phase1"] = self.train_jepa(train_loader)
        results["phase1_time"] = time.time() - t0
        print(f"  Phase 1 complete in {results['phase1_time']:.1f}s")
        self._save_checkpoint("phase1_jepa")

        if val_loader is not None:
            print("  Evaluating...")
            results["phase1_eval"] = self._evaluate(val_loader)

        # Phase 2: World Model Training
        self._clear_gpu_memory()
        print("\n[Phase 2/3] World Model Training...")
        t0 = time.time()
        results["phase2"] = self.train_world_model(train_loader)
        results["phase2_time"] = time.time() - t0
        print(f"  Phase 2 complete in {results['phase2_time']:.1f}s")
        self._save_checkpoint("phase2_world")

        if val_loader is not None:
            print("  Evaluating...")
            results["phase2_eval"] = self._evaluate(val_loader)

        # Phase 3: Policy Distillation
        self._clear_gpu_memory()
        print("\n[Phase 3/3] Policy Distillation...")
        t0 = time.time()
        results["phase3"] = self.distill_policy(train_loader)
        results["phase3_time"] = time.time() - t0
        print(f"  Phase 3 complete in {results['phase3_time']:.1f}s")
        self._save_checkpoint("phase3_final")

        if val_loader is not None:
            print("  Final evaluation...")
            results["final_eval"] = self._evaluate(val_loader)

        total_time = sum(
            results.get(f"phase{i}_time", 0) for i in range(1, 4)
        )
        print("\n" + "=" * 60)
        print(f"Training complete in {total_time:.1f}s")
        print("=" * 60)

        return results

    # ======================================================================
    #  Evaluation
    # ======================================================================

    @torch.no_grad()
    def _evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Compute validation metrics.

        Runs the model in eval mode on *val_loader* and returns
        average losses and position statistics.

        Parameters
        ----------
        val_loader : iterable
            Yields the same batch format as the training loader.

        Returns
        -------
        dict
            Validation metrics: ``val_jepa_loss``, ``val_hyp_loss``,
            ``val_total_loss``, ``avg_position_magnitude``.
        """
        self.model.eval()
        total_jepa = 0.0
        total_hyp = 0.0
        total_loss = 0.0
        total_pos_mag = 0.0
        n = 0

        for batch in val_loader:
            context, target, target_pos = self._unpack_jepa_batch(batch)

            out = self.model(context, target, target_pos)
            loss = (
                out["jepa_loss"]
                + self.config.hyp_weight * out["hyperbolic_loss"]
            )

            total_jepa += out["jepa_loss"].item()
            total_hyp += out["hyperbolic_loss"].item()
            total_loss += loss.item()
            total_pos_mag += out["positions"].abs().mean().item()
            n += 1

        self.model.train()

        if n == 0:
            return {"val_total_loss": float("nan")}

        return {
            "val_jepa_loss": total_jepa / n,
            "val_hyp_loss": total_hyp / n,
            "val_total_loss": total_loss / n,
            "avg_position_magnitude": total_pos_mag / n,
        }

    # ======================================================================
    #  Checkpointing
    # ======================================================================

    def _save_checkpoint(self, tag: str) -> str:
        """Save full training state to disk.

        Saves model weights, all optimizer states, global step, config,
        and replay buffer size.

        Parameters
        ----------
        tag : str
            Identifier for this checkpoint (used in filename).

        Returns
        -------
        str
            Path to the saved checkpoint file.
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"nexus_{tag}.pt"

        state = {
            "model_state_dict": self.model.state_dict(),
            "jepa_optimizer": self.jepa_optimizer.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "config": {
                k: v
                for k, v in self.config.__dict__.items()
                if not k.startswith("_")
            },
            "replay_buffer_size": len(self.replay_buffer),
        }
        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")
        return str(path)

    def _load_checkpoint(self, path: str) -> int:
        """Resume training from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Returns
        -------
        int
            The global step at which training was saved.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.jepa_optimizer.load_state_dict(state["jepa_optimizer"])
        self.world_optimizer.load_state_dict(state["world_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = state.get("global_step", 0)
        print(f"  Checkpoint loaded: {path}  (step {self.global_step})")
        return self.global_step

    # ======================================================================
    #  Private Helpers
    # ======================================================================

    def _get_ema_decay(self, step: int, total_steps: int) -> float:
        """Compute EMA decay with linear warmup.

        Linearly interpolates from ``ema_schedule[0]`` at step 0 to
        ``ema_schedule[1]`` at *total_steps*.

        Parameters
        ----------
        step : int
            Current training step.
        total_steps : int
            Total number of training steps.

        Returns
        -------
        float
            EMA decay value in [ema_schedule[0], ema_schedule[1]].
        """
        low, high = self.config.ema_schedule
        progress = min(step / max(total_steps, 1), 1.0)
        return low + (high - low) * progress

    def _collect_transitions(self, dataloader: Any) -> None:
        """Collect world model training transitions from data.

        Runs the current (frozen) encoder and policy on the dataloader
        to produce ``(z, a, r, z_next, done)`` tuples for the replay
        buffer. Rewards are computed as simple log returns when not
        provided explicitly.

        Parameters
        ----------
        dataloader : iterable
            Yields batches containing market data.
        """
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                context = self._extract_context(batch)

                # Encode full sequence
                z_seq = self.model.encode_sequence(context)  # (B, L, d_latent)
                B, L, d_latent = z_seq.shape

                if L < 2:
                    continue

                # Generate actions via the current policy
                for t in range(L - 1):
                    z_t = z_seq[:, t, :]        # (B, d_latent)
                    z_tp1 = z_seq[:, t + 1, :]  # (B, d_latent)

                    # Action: use policy head
                    a_t = (
                        self.model.policy_head(z_t)
                        * self.model.cfg.max_position
                    )

                    # Reward: proxy from latent distance
                    # (smaller distance to next state = more predictable = positive)
                    r_t = -F.mse_loss(
                        z_t, z_tp1, reduction="none"
                    ).mean(dim=-1, keepdim=True)

                    # Extract per-sample rewards if available
                    if isinstance(batch, dict) and "rewards" in batch:
                        rewards = batch["rewards"].to(self.device)
                        if rewards.dim() == 2 and rewards.size(1) > t:
                            r_t = rewards[:, t : t + 1]

                    # Done: last timestep in window
                    done_t = torch.zeros(B, 1, device=self.device)
                    if t == L - 2:
                        done_t = torch.ones(B, 1, device=self.device)

                    self.replay_buffer.add(z_t, a_t, r_t, z_tp1, done_t)

        self.model.train()

    def _unpack_jepa_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize batch format to (context, target, target_positions).

        Handles both tuple and dict batch formats.

        Parameters
        ----------
        batch : tuple or dict
            Either ``(context, target, target_positions)`` or a dict
            with keys ``"context"``, ``"target"``, ``"target_positions"``.

        Returns
        -------
        tuple of torch.Tensor
            ``(context, target, target_positions)`` moved to device.
        """
        if isinstance(batch, dict):
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            target_pos = batch["target_positions"].to(self.device)
        elif isinstance(batch, (tuple, list)):
            context = batch[0].to(self.device)
            target = batch[1].to(self.device)
            target_pos = batch[2].to(self.device)
        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch)}. "
                "Expected tuple, list, or dict."
            )
        return context, target, target_pos

    def _extract_context(self, batch: Any) -> torch.Tensor:
        """Extract context tensor from a heterogeneous batch.

        Parameters
        ----------
        batch : tuple, list, dict, or Tensor
            The training batch.

        Returns
        -------
        torch.Tensor
            Context data on device, shape (B, L, n_features).
        """
        if isinstance(batch, dict):
            return batch["context"].to(self.device)
        elif isinstance(batch, (tuple, list)):
            return batch[0].to(self.device)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch)}. "
                "Expected tuple, list, dict, or Tensor."
            )

    @staticmethod
    def _clear_gpu_memory() -> None:
        """Free unreferenced GPU memory between training phases.

        Must call ``gc.collect()`` before ``torch.cuda.empty_cache()``
        to break circular references (optimizer -> params -> module).
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _log(
        phase: str,
        step: int,
        total: int,
        loss: float,
        extras: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print formatted training progress.

        Parameters
        ----------
        phase : str
            Phase name (``"JEPA"``, ``"World"``, ``"Distill"``).
        step : int
            Current step within this phase.
        total : int
            Estimated total steps for this phase.
        loss : float
            Current loss value.
        extras : dict, optional
            Additional metrics to display.
        """
        pct = 100.0 * step / max(total, 1)
        msg = f"  [{phase}] step {step}/{total} ({pct:.0f}%)  loss={loss:.6f}"
        if extras:
            extra_str = "  ".join(f"{k}={v:.5f}" for k, v in extras.items())
            msg += f"  {extra_str}"
        print(msg)
