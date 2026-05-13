"""
Main agent entry point for the competition (V2 — per-planet LLM strategy).

Wraps the DQN-based RL agent with a threaded V2 LLM strategy provider.
The V2 LLM assigns individual commands (A/P/E/N) to each planet based
on frontline/rear analysis.
"""

import os
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, GameParams, Player, Action
from rl_agent.dqn_agent import DQNAgent
from rl_agent.state_encoder import LETTER_TO_VEC, DEFAULT_LLM_VEC
from v2_my_agent_llm import LLMAgent as V2LLMAgent


class MyPythonAgent(PlanetWarsPlayer):
    """Competition agent: DQN RL with threaded V2 per-planet LLM strategy."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.dqn_agent = DQNAgent(checkpoint_path=checkpoint_path)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[MyPythonAgent] WARNING: OPENROUTER_API_KEY not set, LLM disabled")
            self.llm_agent = None
        else:
            self.llm_agent = V2LLMAgent(
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
        """Pull latest per-planet strategy from LLM and convert to vectors."""
        if self.llm_agent is None:
            return
        with self._strategy_lock:
            raw_strategy = dict(self.llm_agent.current_strategy)

        # V2 format: {planet_id: 'A'} → convert to {planet_id: [1,0,0,0]}
        strategy_dict = {}
        for pid, val in raw_strategy.items():
            if isinstance(val, str):
                strategy_dict[pid] = LETTER_TO_VEC.get(val, DEFAULT_LLM_VEC)
            else:
                strategy_dict[pid] = val

        self.dqn_agent.set_llm_strategy(strategy_dict)

    def get_action(self, game_state: GameState) -> Action:
        self.latest_state = game_state
        self._sync_llm_strategy()
        return self.dqn_agent.get_action(game_state)

    def get_agent_type(self) -> str:
        return "LLM+DQN Hybrid Agent v2.0 (per-planet)"