"""
DQN neural network for Pairwise Planet Wars agent.

A small MLP that maps a 23-dim feature vector (source-target pair encoding)
to Q-values for 5 discrete actions (ship fractions: 0%, 25%, 50%, 75%, 100%).
"""

import torch
import torch.nn as nn

from rl_agent.state_encoder import FEATURE_DIM
from rl_agent.action_decoder import NUM_ACTIONS


class DQNNetwork(nn.Module):
    """
    Simple MLP for pairwise Q-value estimation.

    Architecture: Input(23) → 128 → ReLU → 64 → ReLU → 5
    Inference time: <1ms for a batch of 900 vectors on CPU.
    """

    def __init__(
        self,
        state_dim: int = FEATURE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden1: int = 128,
        hidden2: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Q-values tensor of shape (batch_size, action_dim) or (action_dim,).
        """
        return self.net(x)
