"""
Main agent entry point for the competition.

Wraps the DQN-based RL agent and provides the interface expected by
the game runner and competition server.

The LLM strategy integration layer (separate team) will call
set_llm_strategy() on the inner DQN agent periodically.
"""

from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, GameParams, Player, Action
from rl_agent.dqn_agent import DQNAgent


class MyPythonAgent(PlanetWarsPlayer):
    """Competition agent: DQN RL with LLM strategy support."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.dqn_agent = DQNAgent(checkpoint_path=checkpoint_path)

    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = None,
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        self.dqn_agent.prepare_to_play_as(player, params, opponent)
        return self.get_agent_type()

    def get_action(self, game_state: GameState) -> Action:
        return self.dqn_agent.get_action(game_state)

    def get_agent_type(self) -> str:
        return "LLM+DQN Hybrid Agent v1.0"