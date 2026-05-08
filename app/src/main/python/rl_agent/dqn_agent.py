"""
DQN Agent for Planet Wars competition.

This is the 'battle-ready' agent that loads pre-trained DQN weights
and plays using the pairwise Q-evaluation architecture.

Inherits from PlanetWarsPlayer so it integrates directly with
the existing game runner infrastructure.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, GameParams, Player, Action

from rl_agent.state_encoder import StateEncoder, FEATURE_DIM
from rl_agent.action_decoder import ActionDecoder, NUM_ACTIONS
from rl_agent.dqn_network import DQNNetwork


# Default checkpoint path (relative to PYTHONPATH root)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "checkpoints", "dqn_latest.pt"
)


class DQNAgent(PlanetWarsPlayer):
    """
    Pre-trained DQN agent for Planet Wars.

    On each tick:
      1. Encodes all valid (source, target) pairs.
      2. Evaluates Q-values for all pairs in a single batched forward pass.
      3. Selects the pair + fraction with the highest Q-value.
      4. Returns the corresponding Action.

    The LLM strategy vector can be updated externally via set_llm_strategy().
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._network: Optional[DQNNetwork] = None
        self._encoder: Optional[StateEncoder] = None
        self._decoder = ActionDecoder()

        # LLM strategy — updated externally by the integration layer
        self._llm_strategy: Dict[int, List[float]] = {}

    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = None,
    ) -> str:
        """Initialize encoder and load network weights."""
        super().prepare_to_play_as(player, params, opponent)
        self._encoder = StateEncoder(params)

        # Load or re-load the network
        self._network = DQNNetwork()
        if os.path.exists(self._checkpoint_path):
            self._network.load_state_dict(
                torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
            )
            print(f"[DQNAgent] Loaded weights from {self._checkpoint_path}")
        else:
            print(
                f"[DQNAgent] WARNING: No checkpoint at {self._checkpoint_path}, "
                f"using random weights!"
            )
        self._network.eval()

        return self.get_agent_type()

    def get_action(self, game_state: GameState) -> Action:
        """Select the best action via pairwise Q-evaluation."""
        if self._encoder is None or self._network is None:
            return Action.do_nothing()

        # Encode all valid (source, target) pairs
        features, pair_info = self._encoder.encode_all_pairs(
            game_state, self.player, self._llm_strategy
        )

        if len(pair_info) == 0:
            return Action.do_nothing()

        # Batched forward pass
        with torch.no_grad():
            q_values = self._network(torch.FloatTensor(features))  # (N, 5)

        # Find global maximum Q-value
        flat_idx = q_values.argmax().item()
        pair_idx = flat_idx // NUM_ACTIONS
        action_idx = flat_idx % NUM_ACTIONS

        # Decode to Action
        source, target = pair_info[pair_idx]
        action = self._decoder.decode(action_idx, source, target, self.player)

        return action

    def set_llm_strategy(self, strategy: Dict[int, List[float]]) -> None:
        """
        Update the LLM strategy vectors.

        Called by the integration layer approximately every 5 seconds.

        Args:
            strategy: Mapping from planet_id to [p_accumulate, p_attack_enemy,
                      p_transfer_ally, p_attack_neutral].
        """
        self._llm_strategy = strategy

    def get_agent_type(self) -> str:
        return "DQN Pairwise RL Agent v1.0"