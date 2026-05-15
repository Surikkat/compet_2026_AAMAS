import time
import json
import re
import threading
from typing import Optional, Dict
from openai import OpenAI
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

class PureLLMAgent(PlanetWarsPlayer):
    """Асинхронный LLM-агент: думает в фоне, действует мгновенно."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Кэш последнего действия от LLM
        self._cached_action: Optional[Action] = None
        self._lock = threading.Lock()
        
        # Состояние для фонового потока
        self._latest_state: Optional[GameState] = None
        self._state_lock = threading.Lock()
        
        # Статистика
        self._llm_calls = 0
        self._cache_hits = 0
        self._fallback_calls = 0
        self._action_times: list = []
        
        # Поток запустим в prepare_to_play_as
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.prompt_template = """You are playing a Planet Wars RTS game as {player}.
        
YOUR PLANETS:
{my_planets}

TARGETS (enemy or neutral):
{targets}

Choose ONE action: send ships from one of your planets to one target.
Return JSON: {{"source": planet_id, "target": planet_id, "fraction": 0.0-1.0}}

Rules:
- Combat is 1:1
- Neutrals don't produce until captured
- Never send all ships from a planet (leave at least 1 for defense)
- Prioritize capturing neutrals early, attacking enemy weak points later

Example: {{"source": 3, "target": 7, "fraction": 0.5}}
Output ONLY JSON, no other text."""
    
    def prepare_to_play_as(
        self, player: Player, params: GameParams, opponent: Optional[str] = None
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        # Запускаем поток только когда player установлен
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._llm_loop, daemon=True)
            self._thread.start()
            print(f"[LLM] Background thread started for {player}")
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        t0 = time.time()
        
        # Обновляем состояние для фона
        with self._state_lock:
            self._latest_state = game_state
        
        # Достаём из кэша
        with self._lock:
            action = self._cached_action
        
        dt = (time.time() - t0) * 1000
        self._action_times.append(dt)
        
        if action is not None:
            self._cache_hits += 1
            return action
        
        # Кэш пуст → fallback
        self._fallback_calls += 1
        return self._fallback_action(game_state)
    
    def _llm_loop(self):
        """Фоновый поток: запрашивает LLM и обновляет кэш."""
        while self._running:
            with self._state_lock:
                state = self._latest_state
            
            if state is None or self.player is None:
                time.sleep(0.05)
                continue
            
            action = self._get_llm_action(state)
            if action is not None:
                with self._lock:
                    self._cached_action = action
                self._llm_calls += 1
                if self._llm_calls <= 5 or self._llm_calls % 20 == 0:
                    print(f"[LLM #{self._llm_calls}] New action cached")
            
            time.sleep(0.05)  # 50ms пауза между запросами
    
    def _get_llm_action(self, game_state: GameState) -> Optional[Action]:
        my_planets = []
        targets = []
        
        for p in game_state.planets:
            info = {
                "id": p.id,
                "ships": int(p.n_ships),
                "growth": round(p.growth_rate, 2),
                "owner": str(p.owner)
            }
            if p.owner == self.player:
                info["incoming_enemy"] = (
                    int(p.transporter.n_ships) if p.transporter and p.transporter.owner != self.player else 0
                )
                my_planets.append(info)
            elif p.owner != self.player:
                targets.append(info)
        
        if not my_planets or not targets:
            return None
        
        prompt = self.prompt_template.format(
            player=str(self.player),
            my_planets=json.dumps(my_planets, indent=2),
            targets=json.dumps(targets, indent=2)
        )
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                timeout=3.0
            )
            
            text = response.choices[0].message.content
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            
            data = json.loads(match.group(0))
            src_id = int(data.get("source", -1))
            tgt_id = int(data.get("target", -1))
            fraction = float(data.get("fraction", 0.5))
            
            source = next((p for p in game_state.planets if p.id == src_id), None)
            if not source or source.owner != self.player or source.n_ships < 2:
                return None
            
            ships = max(1, int(source.n_ships * fraction))
            ships = min(ships, source.n_ships - 1)
            
            return Action(
                playerId=self.player,
                sourcePlanetId=src_id,
                destinationPlanetId=tgt_id,
                numShips=ships
            )
        except Exception as e:
            if self._llm_calls <= 1:
                print(f"[LLM Error] {e}")
            return None
    
    def _fallback_action(self, game_state: GameState) -> Action:
        my_planets = [p for p in game_state.planets if p.owner == self.player and p.n_ships > 1]
        if not my_planets:
            return Action.do_nothing()
        
        source = max(my_planets, key=lambda p: p.n_ships)
        targets = [p for p in game_state.planets if p.owner != self.player]
        if not targets:
            return Action.do_nothing()
        
        target = min(targets, key=lambda p: p.n_ships)
        return Action(
            playerId=self.player,
            sourcePlanetId=source.id,
            destinationPlanetId=target.id,
            numShips=max(1, int(source.n_ships * 0.5))
        )
    
    def get_agent_type(self) -> str:
        return "Pure LLM Agent v2.0"
    
    def get_stats(self) -> dict:
        total = len(self._action_times)
        return {
            "llm_calls": self._llm_calls,
            "cache_hits": self._cache_hits,
            "fallback_calls": self._fallback_calls,
            "total_actions": total,
            "avg_action_time_ms": sum(self._action_times) / total if total else 0,
            "max_action_time_ms": max(self._action_times) if self._action_times else 0,
            "under_50ms_pct": sum(1 for t in self._action_times if t < 50) / total * 100 if total else 0
        }


