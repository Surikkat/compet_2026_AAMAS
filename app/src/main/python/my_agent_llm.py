import time
import json
import re
import threading
from typing import Optional, Dict, Tuple
from openai import OpenAI
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

class PureLLMAgent(PlanetWarsPlayer):
    """Агент с проверенным fallback. LLM опционально."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", use_llm: bool = False):
        super().__init__()
        self.use_llm = use_llm
        self.client = OpenAI(api_key=api_key, base_url=base_url) if use_llm else None
        
        self._llm_cache: Dict[int, Tuple[int, float]] = {}
        self._lock = threading.Lock()
        self._latest_state: Optional[GameState] = None
        self._state_lock = threading.Lock()
        
        self._llm_calls = 0
        self._fallback_calls = 0
        self._action_times: list = []
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        if use_llm:
            self.prompt_template = """You are playing a Planet Wars RTS game as {player}.
            
YOUR PLANETS:
{my_planets}

TARGETS (enemy or neutral):
{targets}

For EACH of your planets, choose a target and fraction of ships to send.
Return JSON: {{"planet_id": ["target_id", fraction], ...}}
Use 0.0 to do nothing from that planet.

Rules:
- Combat is 1:1, neutrals don't produce
- Leave at least 1 ship on each planet for defense
- Prioritize capturing neutrals early
- Attack enemy planets when you have numerical advantage

Example: {{"3": [7, 0.5], "5": [12, 0.3], "8": [8, 0.0]}}
Output ONLY JSON, no other text."""
    
    def prepare_to_play_as(
        self, player: Player, params: GameParams, opponent: Optional[str] = None
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        if self.use_llm and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._llm_loop, daemon=True)
            self._thread.start()
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        t0 = time.time()
        
        # Обновляем состояние
        with self._state_lock:
            self._latest_state = game_state
        
        # Всегда используем проверенный fallback
        action = self._fallback_action(game_state)
        self._fallback_calls += 1
        
        dt = (time.time() - t0) * 1000
        self._action_times.append(dt)
        
        return action
    
    def _llm_loop(self):
        while self._running:
            with self._state_lock:
                state = self._latest_state
            if state and self.player:
                actions = self._get_llm_actions(state)
                if actions:
                    with self._lock:
                        self._llm_cache = actions
                    self._llm_calls += 1
            time.sleep(0.5)
    
    def _get_llm_actions(self, game_state: GameState) -> Dict[int, Tuple[int, float]]:
        if not self.client:
            return {}
        # ... (тот же код запроса к LLM)
        return {}
    
    def _fallback_action(self, game_state: GameState) -> Action:
        """Проверенная стратегия: атаковать слабейшую цель сильнейшей планетой."""
        my_planets = [p for p in game_state.planets if p.owner == self.player and p.n_ships > 1]
        if not my_planets:
            return Action.do_nothing()
        
        source = max(my_planets, key=lambda p: p.n_ships)
        targets = [p for p in game_state.planets if p.owner != self.player]
        if not targets:
            return Action.do_nothing()
        
        # Атакуем самую слабую цель
        target = min(targets, key=lambda p: p.n_ships)
        return Action(
            playerId=self.player,
            sourcePlanetId=source.id,
            destinationPlanetId=target.id,
            numShips=max(1, int(source.n_ships * 0.5))
        )
    
    def get_agent_type(self) -> str:
        return "Pure LLM Agent v4.0 (fallback-only)"
    
    def get_stats(self) -> dict:
        total = len(self._action_times)
        return {
            "llm_calls": self._llm_calls,
            "fallback_calls": self._fallback_calls,
            "total_actions": total,
            "avg_action_time_ms": sum(self._action_times) / total if total else 0,
            "max_action_time_ms": max(self._action_times) if self._action_times else 0,
            "under_50ms_pct": sum(1 for t in self._action_times if t < 50) / total * 100 if total else 0
        }

