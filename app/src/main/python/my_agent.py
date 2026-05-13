"""
Main agent entry point for the competition.

Wraps the DQN-based RL agent with a threaded LLM strategy provider.
The LLM runs in a background daemon thread and updates strategy every ~5s.
The DQN agent runs synchronously within the 50ms tick budget.
"""

import os
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, GameParams, Player, Action
from rl_agent.dqn_agent import DQNAgent
from my_agent_llm import LLMAgent


class MyPythonAgent(PlanetWarsPlayer):
    """Competition agent: DQN RL with threaded LLM strategy."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.dqn_agent = DQNAgent(checkpoint_path=checkpoint_path)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[MyPythonAgent] WARNING: OPENROUTER_API_KEY not set, LLM disabled")
            self.llm_agent = None
        else:
            self.llm_agent = LLMAgent(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        self.latest_state = None
        self._strategy_lock = threading.Lock()
        self._llm_thread_started = False

    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = None,
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        self.dqn_agent.prepare_to_play_as(player, params, opponent)

        # Start LLM thread once
        if self.llm_agent and not self._llm_thread_started:
            self._llm_thread_started = True
            self.llm_thread = threading.Thread(
                target=self.llm_agent.run,
                args=(
                    lambda: self.latest_state,
                    lambda: getattr(self, "player", None),
                ),
                daemon=True,
            )
            self.llm_thread.start()

        return self.get_agent_type()

    def _sync_llm_strategy(self) -> None:
        """Pull latest strategy from LLM thread and push to DQN agent."""
        if self.llm_agent is None:
            return
        with self._strategy_lock:
            strategy_vector = list(self.llm_agent.current_strategy)
        strategy_dict = {}
        if self.latest_state:
            for planet in self.latest_state.planets:
                strategy_dict[planet.id] = strategy_vector
        self.dqn_agent.set_llm_strategy(strategy_dict)

    def get_action(self, game_state: GameState) -> Action:
        self.latest_state = game_state
        self._sync_llm_strategy()
        return self.dqn_agent.get_action(game_state)

    def get_agent_type(self) -> str:
        return "LLM+DQN Hybrid Agent v1.0"