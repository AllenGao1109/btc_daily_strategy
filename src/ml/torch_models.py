"""Small PyTorch models for daily BTC signals, running on Apple MPS when available.

Given the tiny sample size (~3.4k daily rows), models here are deliberately small
and heavily regularized (dropout + weight decay). The goal is a usable directional
signal, not a high-capacity function approximator that memorizes the series.

Each model exposes the minimal fit/predict interface used by the walk-forward
harness:
    model.fit(X_train: np.ndarray, y_train: np.ndarray) -> None
    model.predict(X: np.ndarray) -> np.ndarray   # one value per row
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available torch device (MPS on Apple Silicon, else CPU)."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed torch + numpy for reproducible training."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class _MLP(nn.Module):
    """A small dropout-regularized MLP trunk with a single output head."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLP:
    """MLP wrapper for regression (forward return) or classification (up/down).

    Args:
        task: ``"regression"`` (predict forward return) or ``"classification"``
            (predict P(up) via logits; predict() returns probability-0.5 in
            [-0.5, 0.5] so the sign is the directional call).
        hidden: Hidden layer sizes.
        dropout: Dropout probability.
        lr: Adam learning rate.
        weight_decay: L2 regularization (key defense against overfitting).
        epochs: Training epochs.
        seed: RNG seed.
        prefer_mps: Use MPS if available.
    """

    def __init__(
        self,
        task: str = "classification",
        hidden: tuple[int, ...] = (32, 16),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 80,
        seed: int = 0,
        prefer_mps: bool = True,
    ):
        if task not in {"regression", "classification"}:
            raise ValueError(f"Unknown task: {task!r}")
        self.task = task
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.device = get_device(prefer_mps)
        self.model: _MLP | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        """Train on standardized features X and targets y (forward returns)."""
        set_seed(self.seed)
        self.model = _MLP(X.shape[1], self.hidden, self.dropout).to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.task == "classification":
            target = torch.tensor(
                (y > 0).astype(np.float32), dtype=torch.float32, device=self.device
            )
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            target = torch.tensor(y.astype(np.float32), device=self.device)
            loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.model(xb)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict per-row outputs (forward return, or P(up)-0.5 for classification)."""
        assert self.model is not None, "Model not fitted."
        self.model.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = self.model(xb)
        if self.task == "classification":
            out = torch.sigmoid(out) - 0.5
        return out.detach().cpu().numpy()
