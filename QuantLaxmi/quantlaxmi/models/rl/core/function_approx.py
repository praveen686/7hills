"""Function Approximation for Reinforcement Learning.

Implements the function approximation framework from:
  "Foundations of Reinforcement Learning with Applications in Finance"
  (Rao & Jelvis, Stanford CME 241), Chapter 6.

Chapter 6: Function Approximation and Approximate Dynamic Programming
  §6.1: The need for function approximation in large/continuous state spaces
  §6.2: Linear function approximation with feature vectors
  §6.3: Deep neural network function approximation
  §6.4: Approximate dynamic programming (ADP) framework

The key insight is that in large or continuous state spaces, we cannot
store V(s) or Q(s,a) for every state.  Instead, we parameterise these
functions:
  - Linear FA:  v̂(s, w) = w^T · φ(s)
  - DNN FA:     v̂(s, θ) = f_θ(φ(s))   where f_θ is a neural network

Both share the FunctionApprox[X] interface: evaluate, update, solve.
"""
from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    "FunctionApprox",
    "Tabular",
    "LinearFunctionApprox",
    "DNNSpec",
    "DNNApprox",
    "AdamGradient",
]

X = TypeVar("X")


# ---------------------------------------------------------------------------
# Adam gradient configuration
# ---------------------------------------------------------------------------


class AdamGradient(NamedTuple):
    """Configuration for Adam optimiser (Kingma & Ba, 2014).

    Used by LinearFunctionApprox and DNNApprox.
    """

    learning_rate: float = 1e-3
    decay1: float = 0.9   # β₁
    decay2: float = 0.999  # β₂


# ---------------------------------------------------------------------------
# Abstract Function Approximator  (Ch 6, §6.1)
# ---------------------------------------------------------------------------


class FunctionApprox(ABC, Generic[X]):
    """Abstract function approximator: X → float.

    This is the core abstraction for generalising value functions
    across states.  Instead of storing V(s) for every s, we learn
    a parameterised function v̂(s, w) that generalises.

    Reference: Ch 6, §6.1 — "The function approximation framework
    parameterises the value function as v̂(s; w) where w are learnable
    weights.  The objective is to minimise the Mean Squared Value Error
    (MSVE): overline{VE}(w) = Σ_s μ(s)·[v^π(s) - v̂(s;w)]²"

    Three backends:
      - Tabular: direct dict lookup (no generalisation)
      - Linear:  v̂(s;w) = w^T·φ(s) with feature extraction
      - DNN:     v̂(s;θ) = f_θ(φ(s)) with neural network
    """

    @abstractmethod
    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        """Compute v̂(x) for a batch of inputs.

        Args:
            x_values: Iterable of input points.

        Returns:
            1-D numpy array of predicted values.
        """

    @abstractmethod
    def update(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> FunctionApprox[X]:
        """Perform one (stochastic) gradient update.

        Moves the parameters w in the direction that reduces the loss
        on the given (x, y) pairs:
          w ← w - α · ∇_w L(w)
        where L(w) = Σ_i (y_i - v̂(x_i; w))²

        Returns a new FunctionApprox with updated parameters.
        (Functional style: original is NOT modified.)
        """

    @abstractmethod
    def solve(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> FunctionApprox[X]:
        """Find the best-fit parameters in closed form (if possible).

        For linear FA: w* = (Φ^T Φ + λI)^{-1} Φ^T y
        For DNN: iterative solve via multiple gradient steps.
        """

    def __call__(self, x: X) -> float:
        """Convenience: evaluate a single point."""
        result = self.evaluate([x])
        return float(result[0])

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        """Check if this approximation is within tolerance of another.

        Used for convergence checks in DP and RL algorithms.
        """
        return False  # Subclasses override


# ---------------------------------------------------------------------------
# Tabular Function Approximation  (Ch 6, §6.1 — baseline)
# ---------------------------------------------------------------------------


class Tabular(FunctionApprox[X]):
    """Tabular function approximation — stores values in a dictionary.

    This is the degenerate case of function approximation with no
    generalisation: each state has its own independent parameter.
    Equivalent to having |S| features, each an indicator for one state.

    Update rule:
      V(x) ← V(x) + α · (y - V(x))

    Reference: Ch 6, §6.1 — "Tabular methods can be viewed as a special
    case of linear FA with indicator features."
    """

    def __init__(
        self,
        values_map: dict[X, float] | None = None,
        count_map: dict[X, int] | None = None,
        learning_rate: float = 0.1,
    ) -> None:
        self.values_map: dict[X, float] = dict(values_map or {})
        self.count_map: dict[X, int] = dict(count_map or {})
        self.learning_rate = learning_rate

    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        xs = list(x_values)
        return np.array([self.values_map.get(x, 0.0) for x in xs])

    def update(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> Tabular[X]:
        """SGD update: V(x) ← V(x) + α·(y - V(x))."""
        new_values = dict(self.values_map)
        new_counts = dict(self.count_map)

        for x, y in xy_values:
            old_v = new_values.get(x, 0.0)
            new_values[x] = old_v + self.learning_rate * (y - old_v)
            new_counts[x] = new_counts.get(x, 0) + 1

        return Tabular(
            values_map=new_values,
            count_map=new_counts,
            learning_rate=self.learning_rate,
        )

    def solve(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> Tabular[X]:
        """Direct solve: compute the mean target for each x."""
        sums: dict[X, float] = {}
        counts: dict[X, int] = {}
        for x, y in xy_values:
            sums[x] = sums.get(x, 0.0) + y
            counts[x] = counts.get(x, 0) + 1
        new_values = {x: sums[x] / counts[x] for x in sums}
        return Tabular(
            values_map=new_values,
            count_map=counts,
            learning_rate=self.learning_rate,
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if not isinstance(other, Tabular):
            return False
        all_keys = set(self.values_map.keys()) | set(other.values_map.keys())
        for k in all_keys:
            if abs(self.values_map.get(k, 0.0) - other.values_map.get(k, 0.0)) > tolerance:
                return False
        return True

    def __repr__(self) -> str:
        return f"Tabular({len(self.values_map)} entries, α={self.learning_rate})"


# ---------------------------------------------------------------------------
# Linear Function Approximation  (Ch 6, §6.2)
# ---------------------------------------------------------------------------


class LinearFunctionApprox(FunctionApprox[X]):
    """Linear function approximation with feature extraction.

    Model: v̂(s; w) = w^T · φ(s) = Σ_i w_i · φ_i(s)

    Where φ: X → R^d is a vector of d feature functions.

    The key property of linear FA is that the gradient ∇_w v̂ = φ(s),
    which makes the SGD update:
      w ← w + α · (y - w^T·φ(x)) · φ(x)

    This is equivalent to the LMS (Widrow-Hoff) rule.

    Direct solve (ordinary least squares):
      w* = (Φ^T Φ + λI)^{-1} Φ^T y
    where Φ is the n×d feature matrix and λ is regularisation.

    Reference: Ch 6, §6.2 — "Linear Value Function Approximation."

    Eq (6.3): SGD update  w ← w + α·δ_t·φ(s_t)
    Eq (6.5): Direct solve  w* = (Φ^T Φ)^{-1} Φ^T v
    """

    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        weights: np.ndarray | None = None,
        regularization_coeff: float = 0.0,
        adam_gradient: AdamGradient = AdamGradient(),
        direct_solve: bool = False,
    ) -> None:
        """
        Args:
            feature_functions: List of d functions φ_i: X → float.
            weights: Weight vector w ∈ R^d. Initialised to zeros if None.
            regularization_coeff: L2 regularisation coefficient λ.
            adam_gradient: Adam optimiser parameters.
            direct_solve: If True, solve() uses least squares rather than SGD.
        """
        self.feature_functions = list(feature_functions)
        self.feature_dim = len(self.feature_functions)
        self.regularization_coeff = regularization_coeff
        self.adam_gradient = adam_gradient
        self.direct_solve_flag = direct_solve

        if weights is not None:
            if weights.shape != (self.feature_dim,):
                raise ValueError(
                    f"weights shape {weights.shape} != expected ({self.feature_dim},)"
                )
            self.weights = weights.copy()
        else:
            self.weights = np.zeros(self.feature_dim, dtype=np.float64)

        # Adam state
        self._adam_m = np.zeros(self.feature_dim, dtype=np.float64)
        self._adam_v = np.zeros(self.feature_dim, dtype=np.float64)
        self._adam_t = 0

    def _features(self, x: X) -> np.ndarray:
        """Compute feature vector φ(x) ∈ R^d."""
        return np.array(
            [f(x) for f in self.feature_functions], dtype=np.float64
        )

    def _feature_matrix(self, xs: list[X]) -> np.ndarray:
        """Compute feature matrix Φ of shape (n, d)."""
        return np.array(
            [[f(x) for f in self.feature_functions] for x in xs],
            dtype=np.float64,
        )

    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        """Forward pass: v̂(x) = w^T · φ(x) for each x."""
        xs = list(x_values)
        if not xs:
            return np.array([], dtype=np.float64)
        Phi = self._feature_matrix(xs)
        return Phi @ self.weights

    def update(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> LinearFunctionApprox[X]:
        """One SGD step with Adam optimiser.

        For each (x, y) pair, the gradient of the squared loss is:
          ∇_w (y - w^T·φ(x))² = -2·(y - w^T·φ(x))·φ(x)

        With L2 regularisation:
          ∇_w [L + λ||w||²] = ∇_w L + 2λ·w

        We use Adam for adaptive learning rates.

        Reference: Ch 6, §6.2, Eq (6.3).
        """
        pairs = list(xy_values)
        if not pairs:
            return self._copy()

        xs = [x for x, _ in pairs]
        ys = np.array([y for _, y in pairs], dtype=np.float64)
        Phi = self._feature_matrix(xs)

        # Predictions
        y_hat = Phi @ self.weights

        # Gradient: -(2/n)·Φ^T·(y - ŷ) + 2λ·w
        n = len(pairs)
        errors = ys - y_hat
        grad = -(2.0 / n) * (Phi.T @ errors) + 2.0 * self.regularization_coeff * self.weights

        # Adam update
        new_t = self._adam_t + 1
        lr = self.adam_gradient.learning_rate
        beta1 = self.adam_gradient.decay1
        beta2 = self.adam_gradient.decay2
        eps = 1e-8

        new_m = beta1 * self._adam_m + (1 - beta1) * grad
        new_v = beta2 * self._adam_v + (1 - beta2) * (grad ** 2)

        m_hat = new_m / (1 - beta1 ** new_t)
        v_hat = new_v / (1 - beta2 ** new_t)

        new_weights = self.weights - lr * m_hat / (np.sqrt(v_hat) + eps)

        result = self._copy()
        result.weights = new_weights
        result._adam_m = new_m
        result._adam_v = new_v
        result._adam_t = new_t
        return result

    def solve(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> LinearFunctionApprox[X]:
        """Direct solve: w* = (Φ^T Φ + λI)^{-1} Φ^T y.

        Ordinary Least Squares (with L2 regularisation = Ridge Regression).

        Reference: Ch 6, §6.2, Eq (6.5) — "The closed-form solution
        to the linear least-squares problem."
        """
        pairs = list(xy_values)
        if not pairs:
            return self._copy()

        xs = [x for x, _ in pairs]
        ys = np.array([y for _, y in pairs], dtype=np.float64)
        Phi = self._feature_matrix(xs)

        # w* = (Φ^T Φ + λI)^{-1} Φ^T y
        A = Phi.T @ Phi + self.regularization_coeff * np.eye(
            self.feature_dim, dtype=np.float64
        )
        b = Phi.T @ ys

        try:
            new_weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse
            new_weights = np.linalg.lstsq(Phi, ys, rcond=None)[0]

        result = self._copy()
        result.weights = new_weights
        return result

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        """Check convergence: max |w_i - w'_i| < tolerance."""
        if not isinstance(other, LinearFunctionApprox):
            return False
        return float(np.max(np.abs(self.weights - other.weights))) < tolerance

    def _copy(self) -> LinearFunctionApprox[X]:
        """Create a shallow copy preserving Adam state."""
        result = LinearFunctionApprox(
            feature_functions=self.feature_functions,
            weights=self.weights.copy(),
            regularization_coeff=self.regularization_coeff,
            adam_gradient=self.adam_gradient,
            direct_solve=self.direct_solve_flag,
        )
        result._adam_m = self._adam_m.copy()
        result._adam_v = self._adam_v.copy()
        result._adam_t = self._adam_t
        return result

    def __repr__(self) -> str:
        return (
            f"LinearFA(d={self.feature_dim}, "
            f"||w||={np.linalg.norm(self.weights):.4f}, "
            f"λ={self.regularization_coeff})"
        )


# ---------------------------------------------------------------------------
# DNN Specification  (Ch 6, §6.3)
# ---------------------------------------------------------------------------


class DNNSpec(NamedTuple):
    """Configuration for a single hidden layer in the DNN.

    Reference: Ch 6, §6.3 — "Deep Neural Network Function Approximation."
    """

    neurons: int
    bias: bool = True
    hidden_activation: str = "relu"  # "relu", "tanh", "elu", "leaky_relu"
    dropout: float = 0.0
    batch_norm: bool = False


def _get_activation(name: str) -> nn.Module:
    """Map activation name string to PyTorch module."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DNNApprox")
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    return activations[name]


# ---------------------------------------------------------------------------
# DNN Function Approximation  (Ch 6, §6.3)
# ---------------------------------------------------------------------------


class DNNApprox(FunctionApprox[X]):
    """Deep Neural Network Function Approximation using PyTorch.

    Model: v̂(s; θ) = f_θ(φ(s))

    Where:
      - φ: X → R^d is a feature extraction function
      - f_θ: R^d → R is a neural network with parameters θ

    The network architecture is configurable via a sequence of DNNSpec
    layers.  Training uses Adam optimiser with MSE loss.

    Reference: Ch 6, §6.3 — "Non-linear (Deep) Function Approximation."

    "When the number of features is large or the relationship between
    features and values is non-linear, we can use a deep neural network
    to approximate the value function.  The DNN takes the feature vector
    φ(s) as input and outputs the predicted value v̂(s; θ)."

    Key equations:
      Loss:    L(θ) = (1/n) Σ_i (y_i - f_θ(φ(x_i)))² + λ·||θ||²
      Update:  θ ← θ - α · ∇_θ L(θ)   (via Adam)

    GPU support: auto-detects CUDA (T4 GPU) for acceleration.
    """

    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: Sequence[DNNSpec],
        learning_rate: float = 1e-3,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        regularization_coeff: float = 0.0,
        device: str = "auto",
    ) -> None:
        """
        Args:
            feature_functions: List of d functions φ_i: X → float.
            dnn_spec: Sequence of DNNSpec defining hidden layers.
            learning_rate: Adam learning rate α.
            adam_betas: Adam (β₁, β₂) parameters.
            regularization_coeff: L2 weight decay coefficient λ.
            device: "auto" (detect GPU), "cuda", or "cpu".
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DNNApprox. "
                "Install with: pip install torch"
            )

        self.feature_functions = list(feature_functions)
        self.feature_dim = len(self.feature_functions)
        self.dnn_spec = list(dnn_spec)
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.regularization_coeff = regularization_coeff

        # Device selection
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Build network
        self.model = self._build_model()
        self.model.to(self.device)

        # Optimiser
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            weight_decay=regularization_coeff,
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

    def _build_model(self) -> nn.Module:
        """Construct the PyTorch neural network from DNNSpec.

        Architecture:
          Input(d) → [Linear → BN? → Activation → Dropout?] × L → Linear(1)
        """
        layers: list[nn.Module] = []
        in_dim = self.feature_dim

        for spec in self.dnn_spec:
            # Linear layer
            layers.append(nn.Linear(in_dim, spec.neurons, bias=spec.bias))

            # Batch normalisation (before activation, per convention)
            if spec.batch_norm:
                layers.append(nn.BatchNorm1d(spec.neurons))

            # Activation
            layers.append(_get_activation(spec.hidden_activation))

            # Dropout (after activation)
            if spec.dropout > 0.0:
                layers.append(nn.Dropout(p=spec.dropout))

            in_dim = spec.neurons

        # Output layer: single scalar value
        layers.append(nn.Linear(in_dim, 1, bias=True))

        return nn.Sequential(*layers)

    def _extract_features(self, x_values: list[X]) -> torch.Tensor:
        """Convert inputs to feature tensor on device.

        Shape: (batch_size, feature_dim)
        """
        features = np.array(
            [[f(x) for f in self.feature_functions] for x in x_values],
            dtype=np.float32,
        )
        return torch.from_numpy(features).to(self.device)

    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        """Forward pass: v̂(x) = f_θ(φ(x)) for each x.

        Returns numpy array on CPU.
        """
        xs = list(x_values)
        if not xs:
            return np.array([], dtype=np.float64)

        self.model.eval()
        with torch.no_grad():
            features = self._extract_features(xs)
            output = self.model(features).squeeze(-1)
            return output.cpu().numpy().astype(np.float64)

    def update(
        self,
        xy_values: Iterable[tuple[X, float]],
    ) -> DNNApprox[X]:
        """One SGD step using Adam optimiser.

        Computes loss = MSE(y, f_θ(φ(x))) and performs backpropagation.

        Reference: Ch 6, §6.3 — "The gradient ∇_θ L(θ) is computed
        via backpropagation and the parameters are updated via Adam."

        Returns a new DNNApprox with updated weights.
        """
        pairs = list(xy_values)
        if not pairs:
            return self._copy()

        xs = [x for x, _ in pairs]
        ys = np.array([y for _, y in pairs], dtype=np.float32)

        self.model.train()
        features = self._extract_features(xs)
        targets = torch.from_numpy(ys).to(self.device)

        # Forward
        predictions = self.model(features).squeeze(-1)
        loss = self.loss_fn(predictions, targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return copy with updated state
        # (We mutate in-place for efficiency but return self for chaining)
        return self

    def solve(
        self,
        xy_values: Iterable[tuple[X, float]],
        num_epochs: int = 1000,
        batch_size: int = 256,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> DNNApprox[X]:
        """Iterative solve: train until convergence on the given data.

        Repeatedly passes through the data for num_epochs or until
        the loss change is below tolerance.

        Args:
            xy_values: Training data (x, y) pairs.
            num_epochs: Maximum training epochs.
            batch_size: Mini-batch size for SGD.
            tolerance: Stop if |loss_{k} - loss_{k-1}| < tolerance.
            verbose: Print loss every 100 epochs.

        Returns:
            Self with trained parameters.
        """
        pairs = list(xy_values)
        if not pairs:
            return self

        xs = [x for x, _ in pairs]
        ys_np = np.array([y for _, y in pairs], dtype=np.float32)

        features = self._extract_features(xs)
        targets = torch.from_numpy(ys_np).to(self.device)

        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        prev_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_features, batch_targets in loader:
                predictions = self.model(batch_features).squeeze(-1)
                loss = self.loss_fn(predictions, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

            if abs(prev_loss - avg_loss) < tolerance:
                if verbose:
                    print(f"  Converged at epoch {epoch+1}")
                break
            prev_loss = avg_loss

        return self

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        """Check if model weights are within tolerance of another DNNApprox."""
        if not isinstance(other, DNNApprox):
            return False

        max_diff = 0.0
        for p1, p2 in zip(self.model.parameters(), other.model.parameters()):
            diff = torch.max(torch.abs(p1.data - p2.data)).item()
            max_diff = max(max_diff, diff)
        return max_diff < tolerance

    def _copy(self) -> DNNApprox[X]:
        """Deep copy of the DNNApprox including model weights."""
        new = DNNApprox(
            feature_functions=self.feature_functions,
            dnn_spec=self.dnn_spec,
            learning_rate=self.learning_rate,
            adam_betas=self.adam_betas,
            regularization_coeff=self.regularization_coeff,
            device=str(self.device),
        )
        new.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        new.optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        return new

    def get_model_summary(self) -> str:
        """Human-readable model architecture summary."""
        lines = [f"DNNApprox (device={self.device})"]
        lines.append(f"  Feature dim: {self.feature_dim}")
        total_params = 0
        for name, param in self.model.named_parameters():
            lines.append(f"  {name}: {list(param.shape)}")
            total_params += param.numel()
        lines.append(f"  Total parameters: {total_params:,}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"DNNApprox(d={self.feature_dim}, "
            f"layers={[s.neurons for s in self.dnn_spec]}, "
            f"params={total_params:,}, "
            f"device={self.device})"
        )
