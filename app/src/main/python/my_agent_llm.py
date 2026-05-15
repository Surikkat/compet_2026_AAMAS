import time
import json
import re
import threading
from typing import Optional, Dict, List
from openai import OpenAI

from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

class PureLLMAgent(PlanetWarsPlayer):
    """
    LLM-only агент. Фоновый поток обновляет действие асинхронно.
    Основной поток возвращает последнее готовое действие < 1 мс.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Кэш действия: {planet_id: (target_id, fraction)}
        self._action_cache: Dict[int, tuple] = {}
        self._lock = threading.Lock()
        
        # Последнее использованное состояние
        self._latest_state: Optional[GameState] = None
        self._state_lock = threading.Lock()
        
        # Статистика
        self._llm_calls = 0
        self._cache_hits = 0
        
        # Фоновый поток
        self._running = True
        self._thread = threading.Thread(target=self._llm_loop, daemon=True)
        self._thread.start()
        
        # Промпт
        self.prompt_template = """You control Player {player} in a Planet Wars RTS game.
        
YOUR PLANETS (send ships FROM these):
{my_planets}

POSSIBLE TARGETS (send ships TO these):
{targets}

RULES:
- You can send from EACH of your planets to ONE target
- Specify fraction of ships to send (0.0 to 1.0)
- 0.0 means "do nothing from this planet"
- Combat is 1:1
- Ships take time to arrive (distance / speed)
- Neutrals don't produce until captured

STRATEGY:
- If enemy has more planets -> defend and expand
- If you have more ships -> attack weakest enemy
- Always capture neutrals when safe
- Never leave a planet empty if enemy is close

OUTPUT ONLY JSON (no text):
{{"source_id": ["target_id", fraction], ...}}

Example: {{"3": [7, 0.5], "5": [12, 0.3], "8": [8, 0.0]}}
means: from planet 3 send 50% ships to 7, from 5 send 30% to 12, from 8 do nothing.
"""
    
    def _parse_state_for_llm(self, game_state: GameState) -> tuple:
        """Возвращает (my_planets_str, targets_str, quick_actions)"""
        my_planets = []
        targets = []
        
        for p in game_state.planets:
            info = {
                "id": p.id,
                "ships": int(p.n_ships),
                "x": round(p.position.x, 1),
                "y": round(p.position.y, 1),
                "growth": round(p.growth_rate, 2),
                "owner": str(p.owner)
            }
            if p.owner == self.player:
                info["incoming_enemy"] = 0
                if p.transporter and p.transporter.owner != self.player:
                    info["incoming_enemy"] = int(p.transporter.n_ships)
                my_planets.append(info)
            elif p.owner != self.player:
                targets.append(info)
        
        return json.dumps(my_planets, indent=2), json.dumps(targets, indent=2)
    
    def _get_llm_action(self, game_state: GameState) -> Dict[int, tuple]:
        """Запросить действие у LLM (вызывается в фоновом потоке)."""
        my_planets, targets = self._parse_state_for_llm(game_state)
        
        prompt = self.prompt_template.format(
            player=str(self.player),
            my_planets=my_planets,
            targets=targets
        )
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
                timeout=3.0  # Не ждать дольше 3 секунд
            )
            
            text = response.choices[0].message.content
            return self._parse_action_response(text, game_state)
        except Exception as e:
            print(f"[LLM Action Error] {e}")
            return {}
    
    def _parse_action_response(self, text: str, game_state: GameState) -> Dict[int, tuple]:
        """Парсим JSON от LLM в словарь действий."""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return {}
            
            data = json.loads(match.group(0))
            result = {}
            
            my_planet_ids = {p.id for p in game_state.planets if p.owner == self.player}
            
            for src_id_str, action in data.items():
                src_id = int(src_id_str)
                if src_id not in my_planet_ids:
                    continue
                
                if isinstance(action, list) and len(action) == 2:
                    target_id = int(action[0])
                    fraction = float(action[1])
                    # Валидация
                    if 0.0 <= fraction <= 1.0 and target_id != src_id:
                        result[src_id] = (target_id, fraction)
            
            return result
        except Exception as e:
            print(f"[Parse Error] {e}")
            return {}
    
    def _llm_loop(self):
        """Фоновый цикл: постоянно запрашивает действия у LLM."""
        while self._running:
            with self._state_lock:
                state = self._latest_state
            
            if state and self.player:
                actions = self._get_llm_action(state)
                if actions:
                    with self._lock:
                        self._action_cache = actions
                    self._llm_calls += 1
            
            time.sleep(0.05)  # 50ms между запросами к LLM
    
    def prepare_to_play_as(
        self,
        player: Player,
        params: GameParams,
        opponent: Optional[str] = None,
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        """Мгновенно вернуть действие из кэша."""
        # Обновляем состояние для фонового потока
        with self._state_lock:
            self._latest_state = game_state
        
        # Пытаемся взять из кэша
        with self._lock:
            cache = dict(self._action_cache)
        
        if not cache:
            # Кэш пуст — fallback: атаковать случайную цель
            return self._fallback_action(game_state)
        
        self._cache_hits += 1
        
        # Найти свою планету с максимальным произведением ships * fraction
        best_action = None
        best_value = -1
        
        for src_id, (target_id, fraction) in cache.items():
            source = next((p for p in game_state.planets if p.id == src_id), None)
            if source and source.owner == self.player and source.n_ships > 0:
                ships_to_send = int(source.n_ships * fraction)
                if ships_to_send > 0:
                    value = ships_to_send * fraction  # Приоритет: много кораблей + большая фракция
                    if value > best_value:
                        best_value = value
                        best_action = Action(
                            playerId=self.player,
                            sourcePlanetId=src_id,
                            destinationPlanetId=target_id,
                            numShips=ships_to_send
                        )
        
        if best_action:
            return best_action
        
        return self._fallback_action(game_state)
    
    def _fallback_action(self, game_state: GameState) -> Action:
        """Запасное действие, если LLM не успела."""
        my_planets = [p for p in game_state.planets if p.owner == self.player and p.n_ships > 1]
        if not my_planets:
            return Action.do_nothing()
        
        source = max(my_planets, key=lambda p: p.n_ships)
        targets = [p for p in game_state.planets if p.owner != self.player]
        if not targets:
            return Action.do_nothing()
        
        # Отправляем половину с самой сильной планеты на самую слабую цель
        target = min(targets, key=lambda p: p.n_ships)
        return Action(
            playerId=self.player,
            sourcePlanetId=source.id,
            destinationPlanetId=target.id,
            numShips=int(source.n_ships * 0.5)
        )
    
    def get_agent_type(self) -> str:
        return "Pure LLM Agent v1.0"
    
    def get_stats(self) -> dict:
        return {
            "llm_calls": self._llm_calls,
            "cache_hits": self._cache_hits
        }




