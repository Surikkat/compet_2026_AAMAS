import time
import json
import re
import threading
from typing import Optional, Dict, List
from openai import OpenAI

from core.game_state import GameState, Action, Player

class LLMAgent:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # Default strategy per planet: [A, P, E, N]
        self.current_strategy: Dict[int, List[float]] = {}

        self.last_update_time = 0.0
        self.update_interval = 4.0
        self.running = True
        self.lock = threading.Lock()

        self.prompt_path = "/home/surikkat/code/comp/planet-wars-rts/app/src/main/python/prompt_en.md"

        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print("[Warning] prompt.md not found! Using default template.")
            self.prompt_template = "State:\n{state_summary}\nOutput JSON with per-planet strategies."

    def _parse_state(self, game_state: GameState, player: Player) -> str:
        my_planets_info = []
        my_ships_on_planet = 0.0
        my_ships_on_flight = 0.0
        my_growth = 0.0
        my_planet_count = 0

        enemy_planet = 0
        enemy_ships_on_planet = 0.0
        enemy_ships_on_flight = 0.0
        enemy_growth = 0.0

        neutral_planets = 0
        neutral_ships_on_planet = 0.0

        # Собираем информацию о планетах
        for p in game_state.planets:
            if p.owner == player:
                my_planet_count += 1
                my_ships_on_planet += p.n_ships
                my_growth += p.growth_rate
                my_planets_info.append({
                    "id": p.id,
                    "ships": int(p.n_ships),
                    "x": round(p.position.x, 1),
                    "y": round(p.position.y, 1),
                    "growth": round(p.growth_rate, 2)
                })
            elif p.owner == player.opponent():
                enemy_planet += 1
                enemy_ships_on_planet += p.n_ships
                enemy_growth += p.growth_rate
            elif p.owner == Player.Neutral:
                neutral_planets += 1
                neutral_ships_on_planet += p.n_ships

            if p.transporter is not None:
                if p.transporter.owner == player:
                    my_ships_on_flight += p.transporter.n_ships
                elif p.transporter.owner == player.opponent():
                    enemy_ships_on_flight += p.transporter.n_ships

        my_total_ships = my_ships_on_planet + my_ships_on_flight
        enemy_total_ships = enemy_ships_on_planet + enemy_ships_on_flight

        return json.dumps({
            "my_planets": my_planets_info,
            "my_total_stats": {
                "planets": my_planet_count,
                "total_ships": int(my_total_ships),
                "ships_on_planet": int(my_ships_on_planet),
                "ships_on_flight": int(my_ships_on_flight),
                "growth": round(my_growth, 2)
            },
            "enemy_stats": {
                "planets": enemy_planet,
                "total_ships": int(enemy_total_ships),
                "ships_on_planet": int(enemy_ships_on_planet),
                "ships_on_flight": int(enemy_ships_on_flight),
                "growth": round(enemy_growth, 2)
            },
            "neutral_stats": {
                "planets": neutral_planets,
                "ships": int(neutral_ships_on_planet)
            }
        }, indent=2)

    def _get_llm_output(self, state_summary: str) -> str:
        prompt = self.prompt_template.format(state_summary=state_summary)
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,  # Increased for dictionary output
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

    def _validate(self, text: str) -> Optional[Dict[int, List[float]]]:
        try:
            # Find JSON object in the response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            
            data = json.loads(match.group(0))
            result = {}
            
            for k, v in data.items():
                planet_id = int(k)
                if isinstance(v, list) and len(v) == 4:
                    total = sum(v)
                    if total > 0:
                        result[planet_id] = [x / total for x in v]
            
            return result if result else None
        except Exception:
            return None

    def run(self, get_state_func, get_player_func):
        while self.running:
            current_time = time.time()
            state = get_state_func()
            my_player = get_player_func()

            if state and my_player:
                if current_time - self.last_update_time >= self.update_interval:
                    summary = self._parse_state(state, my_player)
                    print(f"--- Sending to LLM (per-planet) ---")
                    
                    raw_response = self._get_llm_output(summary)
                    valid_strategies = self._validate(raw_response)

                    if valid_strategies:
                        with self.lock:
                            self.current_strategy = valid_strategies
                        
                        state_data = json.loads(summary)
                        print(f"[LLM State] My planets: {state_data['my_total_stats']['planets']} | Enemy: {state_data['enemy_stats']['planets']}")
                        
                        # Show strategy for each planet
                        for pid, vec in valid_strategies.items():
                            print(f"  Planet {pid}: [A={vec[0]:.2f}, P={vec[1]:.2f}, E={vec[2]:.2f}, N={vec[3]:.2f}]")
                        print()
                    else:
                        # Fallback to evenly distributed
                        if state and my_player:
                            fallback = {}
                            for p in state.planets:
                                if p.owner == my_player:
                                    fallback[p.id] = [0.25, 0.25, 0.25, 0.25]
                            with self.lock:
                                self.current_strategy = fallback
                        print(f"[LLM Update] Invalid output, using fallback.\n")

                    self.last_update_time = time.time()
            time.sleep(0.1)




