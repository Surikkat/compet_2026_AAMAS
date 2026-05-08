"""
Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
supports uniform random sampling for mini-batch training.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer for storing experience tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition.

        Args:
            state: Feature vector of the chosen pair, shape (23,).
            action: Action index (0–4).
            reward: Reward received after this action.
            next_state: Feature vector of the same pair at the next tick.
            done: Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of tensors:
              states:      (batch_size, 23)
              actions:     (batch_size,) long
              rewards:     (batch_size,)
              next_states: (batch_size, 23)
              dones:       (batch_size,) float (1.0 if done)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor([float(d) for d in dones]),
        )

    def __len__(self) -> int:
        return len(self.buffer)