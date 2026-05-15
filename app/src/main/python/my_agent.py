from typing import Optional
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

import os
import threading

from rl_agent.dqn_agent import DQNAgent
from my_agent_llm import LLMAgent

class MyPythonAgent(PlanetWarsPlayer):
    """Гибридный агент: DQN + LLM стратегия (per-planet)"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.dqn_agent = DQNAgent(checkpoint_path=checkpoint_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[Warning] OPENROUTER_API_KEY not found, LLM disabled")
            self.llm_agent = None
        else:
            self.llm_agent = LLMAgent(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )

        self.latest_state = None

        if self.llm_agent:
            self.llm_thread = threading.Thread(
                target=self.llm_agent.run,
                args=(
                    lambda: self.latest_state,
                    lambda: getattr(self, 'player', None)
                ),
                daemon=True
            )
            self.llm_thread.start()

    def _sync_llm_strategy(self) -> None:
        """Передать текущую per-planet стратегию от LLM к DQN."""
        if self.llm_agent is None:
            return
        
        with self.llm_agent.lock:
            strategy_dict = dict(self.llm_agent.current_strategy)
        
        # DQN ожидает словарь {planet_id: [A, P, E, N]}
        self.dqn_agent.set_llm_strategy(strategy_dict)

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
        self.latest_state = game_state
        self._sync_llm_strategy()
        return self.dqn_agent.get_action(game_state)

    def get_agent_type(self) -> str:
        return "LLM+DQN Hybrid Agent v2.0 (per-planet)"
