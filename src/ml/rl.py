"""Lightweight REINFORCE policy-gradient agent for daily BTC position sizing (MPS).

Formulation (leakage-free, fee-aware):
  - State at day t: standardized features through day t, plus the current
    position (so the agent can reason about turnover cost / holding).
  - Action at day t: a target weight from a small discrete set (e.g. {0,1,2} =
    flat / 1x long / 2x long). Long-biased by default, matching BTC's upward
    drift; the 0 action is the risk-off / "step aside" choice.
  - Reward at day t: position_t * return_{t+1} - fee_rate * |Δposition|.
    The fee term makes the agent internalize the 0.5% per-trade cost, so it
    learns NOT to churn — i.e. trade rarely and hold through trends.

Timing: the action chosen from features_t is the position held into day t+1, so
reward uses return_{t+1}. Emitted as raw_signal[t], the backtest engine lags it
one day, reproducing exactly this PnL — no double-lag, no lookahead.

Discipline: train ONLY on the training window, roll out greedily out-of-sample,
then score through the real engine with year-by-year robustness. RL on ~3.4k
daily rows is highly overfit-prone; trust only fold-robust OOS results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .torch_models import get_device, set_seed


class PolicyNet(nn.Module):
    """Small softmax policy over discrete target-weight actions.

    Memoryless (action depends only on features) so the whole trajectory's logits
    can be computed in a single batched forward pass — ~100x faster than a
    per-day Python loop on MPS. Turnover cost is still learned, via the fee term
    in the reward rather than via an explicit position input.
    """

    def __init__(self, n_features: int, n_actions: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


class ReinforceTrader:
    """REINFORCE agent that learns a discrete daily position-sizing policy.

    Args:
        actions: Target weights the agent may choose (e.g. (0.0, 1.0, 2.0)).
        fee_rate: Per-trade fee charged on turnover inside the reward.
        hidden: Policy hidden width.
        dropout: Policy dropout.
        lr: Adam learning rate.
        weight_decay: L2 regularization.
        gamma: Reward discount (links holding decisions).
        entropy_coef: Entropy bonus weight (exploration).
        epochs: Number of full-trajectory policy-gradient updates.
        seed: RNG seed.
    """

    def __init__(
        self,
        actions: tuple[float, ...] = (0.0, 1.0, 2.0),
        fee_rate: float = 0.005,
        hidden: int = 32,
        dropout: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        gamma: float = 0.97,
        entropy_coef: float = 0.01,
        epochs: int = 300,
        seed: int = 0,
        reward_type: str = "pnl",
        vol_coef: float = 5.0,
    ):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.fee_rate = fee_rate
        self.reward_type = reward_type
        self.vol_coef = vol_coef
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.seed = seed
        self.device = get_device()
        self.policy: PolicyNet | None = None
        self.mu = None
        self.sd = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def fit(self, X_train: np.ndarray, ret_next_train: np.ndarray) -> "ReinforceTrader":
        """Train the policy on one window via REINFORCE.

        Args:
            X_train: Feature matrix (rows = days, in order).
            ret_next_train: ``return_{t+1}`` aligned to each row t (the reward's
                forward return). The last row should have a defined next return.
        """
        set_seed(self.seed)
        self.mu = X_train.mean(axis=0)
        self.sd = X_train.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Xs = self._standardize(X_train)
        n, nf = Xs.shape
        n_actions = len(self.actions)
        self.policy = PolicyNet(nf, n_actions, self.hidden, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        rt = torch.tensor(ret_next_train, dtype=torch.float32, device=self.device)
        action_vals = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        gpow = self.gamma ** torch.arange(n, dtype=torch.float32, device=self.device)

        self.policy.train()
        for _ in range(self.epochs):
            logits = self.policy(Xt)  # [n, n_actions] in one batched pass
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()  # [n] - independent per day (memoryless policy)
            w = action_vals[a]  # chosen target weight per day
            # Turnover from consecutive actions (prev starts flat at 0).
            prev_w = torch.cat([torch.zeros(1, device=self.device), w[:-1]])
            pnl = w * rt
            turn_cost = self.fee_rate * torch.abs(w - prev_w)
            if self.reward_type == "logutil":
                # Growth-optimal (Kelly-like): rewards compounding, so the agent
                # is less likely to sit out uptrends. Clamp to avoid log(<=0).
                rewards = torch.log(torch.clamp(1.0 + pnl, min=1e-3)) - turn_cost
            elif self.reward_type == "vol_pen":
                # Risk-adjusted: penalize squared daily PnL (variance proxy).
                rewards = pnl - turn_cost - self.vol_coef * pnl * pnl
            else:  # "pnl"
                rewards = pnl - turn_cost

            # Discounted reward-to-go via reverse cumulative sum, standardized.
            disc = rewards * gpow
            returns = torch.flip(torch.cumsum(torch.flip(disc, [0]), 0), [0]) / gpow
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            logps = dist.log_prob(a)
            loss = -(logps * returns).mean() - self.entropy_coef * dist.entropy().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def policy_weights(self, X: np.ndarray) -> np.ndarray:
        """Greedy roll-out: return the chosen target weight for each row of X.

        Memoryless and deterministic (argmax) — computed in one batched pass.
        """
        assert self.policy is not None, "Agent not fitted."
        self.policy.eval()
        Xt = torch.tensor(self._standardize(X), dtype=torch.float32, device=self.device)
        a = torch.argmax(self.policy(Xt), dim=1).cpu().numpy()
        return self.actions[a]


def rl_signal(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end: pd.Timestamp,
    actions: tuple[float, ...] = (0.0, 1.0, 2.0),
    **agent_kwargs,
) -> pd.Series:
    """Train a REINFORCE agent on data up to ``train_end`` and roll out over all df.

    Args:
        df: Feature frame with ``close`` and ``feature_cols``.
        feature_cols: Model input columns.
        train_end: Last date used for training (everything after is OOS).
        actions: Discrete target-weight action set.
        **agent_kwargs: Forwarded to :class:`ReinforceTrader`.

    Returns:
        Raw target-weight signal aligned to ``df.index`` (NaN where features are
        NaN). The engine lags it one day for execution.
    """
    ret_next = df["close"].astype(float).pct_change().shift(-1)  # return_{t+1}
    data = df[feature_cols].copy()
    data["__rn__"] = ret_next
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X_all = data[feature_cols].to_numpy(dtype=np.float64)
    rn_all = data["__rn__"].to_numpy(dtype=np.float64)
    train_mask = data.index <= train_end

    agent = ReinforceTrader(actions=actions, **agent_kwargs)
    agent.fit(X_all[train_mask], rn_all[train_mask])
    weights = agent.policy_weights(X_all)

    sig = pd.Series(np.nan, index=df.index, name="raw_signal")
    sig.loc[data.index] = weights
    return sig
