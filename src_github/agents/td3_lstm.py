"""
TD3-LSTM agent.

This is a convenience wrapper around `src/agents/td3.py` that enables the
history/LSTM pathway by default and provides a small history buffer helper
for online interaction loops (similar to the reference snippet you pasted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.td3 import TD3, ReplayBuffer


@dataclass
class HistoryBuffer:
    """
    Fixed-length rolling history buffer for LSTM-TD3.

    Stores previous (obs, act) pairs so you can pass them to TD3_LSTM.select_action().
    """

    obs_dim: int
    act_dim: int
    max_hist_len: int

    def __post_init__(self):
        self.obs_dim = int(self.obs_dim)
        self.act_dim = int(self.act_dim)
        self.max_hist_len = int(self.max_hist_len)
        self.reset()

    def reset(self):
        H = max(1, int(self.max_hist_len))
        self._obs = np.zeros((H, self.obs_dim), dtype=np.float32)
        self._act = np.zeros((H, self.act_dim), dtype=np.float32)
        self._len = 0

    @property
    def length(self) -> int:
        return int(self._len)

    def push_obs(self, obs: np.ndarray):
        """Record current observation into history (typically at episode start and each step)."""
        o = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        H = self._obs.shape[0]
        if self._len >= H:
            self._obs[:-1] = self._obs[1:]
            self._obs[-1] = o
        else:
            self._obs[self._len] = o
            self._len += 1

    def push_act(self, act: np.ndarray):
        """Record executed action into history (call after selecting/executing an action)."""
        a = np.asarray(act, dtype=np.float32).reshape(self.act_dim)
        H = self._act.shape[0]
        if self.length <= 0:
            # If action arrives before any obs, just treat it as first entry.
            self._act[0] = a
            self._len = max(self._len, 1)
            return
        # Align action with the most recent obs index.
        idx = min(self._len - 1, H - 1)
        self._act[idx] = a

    def export(self):
        """
        Returns:
          hist_obs: [T, obs_dim]
          hist_act: [T, act_dim]
          hist_len: scalar int
        """
        T = max(1, self._len)
        return self._obs[:T].copy(), self._act[:T].copy(), int(self._len)


class TD3_LSTM(TD3):
    """
    TD3 with an LSTM history encoder enabled.

    Notes
    - This class exists mostly for ergonomics: it forces use_history=True and
      exposes helper methods consistent with the TD3 class.
    - The actual algorithm and networks are implemented in `agents.td3.TD3`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float = 1.0,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        policy_delay: int = 2,
        act_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        max_hist_len: int = 10,
        hist_with_past_act: bool = True,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=act_limit,
            gamma=gamma,
            polyak=polyak,
            pi_lr=pi_lr,
            q_lr=q_lr,
            policy_delay=policy_delay,
            act_noise=act_noise,
            target_noise=target_noise,
            noise_clip=noise_clip,
            use_history=True,
            hist_with_past_act=bool(hist_with_past_act),
            max_hist_len=int(max_hist_len),
        )

    def make_history_buffer(self) -> HistoryBuffer:
        return HistoryBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, max_hist_len=self.max_hist_len)

    def select_action_with_history(
        self,
        obs: np.ndarray,
        history: Optional[HistoryBuffer],
        noise_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convenience: use a `HistoryBuffer` instance to pass history to TD3.select_action().
        """
        if history is None:
            return self.select_action(obs, noise_scale=noise_scale)
        hist_obs, hist_act, hist_len = history.export()
        return self.select_action(obs, noise_scale=noise_scale, hist_obs=hist_obs, hist_act=hist_act, hist_len=np.array([hist_len], dtype=np.float32))


__all__ = ["TD3_LSTM", "ReplayBuffer", "HistoryBuffer"]


