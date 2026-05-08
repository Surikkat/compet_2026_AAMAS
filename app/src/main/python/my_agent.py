from typing import Optional
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

import os
import threading

from rl_agent.dqn_agent import DQNAgent
from my_agent_llm import LLMAgent

class MyPythonAgent(PlanetWarsPlayer):
    """Мой первый Python-агент для Planet Wars"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.dqn_agent = DQNAgent(checkpoint_path=checkpoint_path)

        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("API ключ OPENROUTER_API_KEY не найден в .env")

        self.llm_agent = LLMAgent(
            api_key=api_key, 
            base_url="https://openrouter.ai/api/v1"
        ) 
        self.latest_state = None
        self._strategy_lock = threading.Lock()

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
        """Передать текущую стратегию от LLM к DQN."""
        if self.llm_agent is None:
            return
        with self._strategy_lock:
            # Берём копию стратегии, чтобы не менять во время итерации
            strategy_vector = list(self.llm_agent.current_strategy)
        # DQN ожидает словарь {planet_id: [p_accumulate, p_attack_enemy, ...]}
        # Пока у нас глобальный вектор, поэтому оборачиваем в ожидаемый формат
        strategy_dict = {}
        if self.latest_state:
            for planet in self.latest_state.planets:
                strategy_dict[planet.id] = strategy_vector
        
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
        return "LLM+DQN Hybrid Agent v1.0"
