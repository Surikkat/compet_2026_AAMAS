import time
import json
import re
import random
import threading
from typing import Optional, List
from openai import OpenAI
from google import genai

from core.game_state import GameState, Action, Player
from agents.planet_wars_agent import PlanetWarsPlayer

class LLMAgent:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        # [A, P, E, N] = [Attack, Protect, Expansion, Nothing]
        # Now it's a dictionary {planet_id: [A, P, E, N]}
        self.current_strategy = {} 
        self.last_update_time = 0.0
        self.update_interval = 5.0 # seconds
        self.running = True
        self.lock = threading.Lock()
        self.prompt_path = "/home/maria/phystech/projects/game_planet_wars/compet_2026_AAMAS/app/src/main/python/v2_prompt_en.md"
        # Load prompt template
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print("[Warning] prompt.md not found! Using default template.")
            self.prompt_template = "State:\n{state_summary}\nOutput [A, P, E, N] as floats summing to 1.0."
    
    def _parse_state(self, game_state: GameState, player: Player) -> str:
        my_info = []
        my_planet = 0
        my_total_ships, my_ships_on_planet, my_ships_on_flight = 0.0, 0.0, 0.0
        my_growth = 0.0 

        enemy_info = []
        enemy_planet = 0
        enemy_total_ships, enemy_ships_on_planet, enemy_ships_on_flight = 0.0, 0.0, 0.0
        enemy_growth = 0.0

        neutral_info = []
        neutral_planets = 0
        neutral_ships_on_planet=0.0

        enemy_planets = [p for p in game_state.planets if p.owner == player.opponent()]
        if not enemy_planets: return json.dumps({"status": "game_over"})

        # Находим центр вражеских сил
        avg_enemy_x = sum(p.position.x for p in enemy_planets) / len(enemy_planets)
        avg_enemy_y = sum(p.position.y for p in enemy_planets) / len(enemy_planets)

        # Считаем среднее расстояние от центра врага до всех наших планет
        my_planets = [p for p in game_state.planets if p.owner == player]
        if not my_planets: return "{}"

        distances = [((p.position.x - avg_enemy_x)**2 + (p.position.y - avg_enemy_y)**2)**0.5 for p in my_planets]
        threshold = sum(distances) / len(distances)

        for p in game_state.planets:
            p_data = {
                "id": p.id,
                "ships": int(p.n_ships),
                "growth": round(p.growth_rate, 2),
                "busy": p.transporter is not None,
            }

            if p.transporter is not None:
                if p.transporter.owner == player:
                    my_ships_on_flight += p.transporter.n_ships
                elif p.transporter.owner == player.opponent():
                    enemy_ships_on_flight += p.transporter.n_ships

            if p.owner == player:
                my_planet += 1
                my_ships_on_planet += p.n_ships
                my_growth += p.growth_rate
                p_data["owner"] = "me"
                dist = ((p.position.x - avg_enemy_x)**2 + (p.position.y - avg_enemy_y)**2)**0.5
                p_data["zone"] = "frontline" if dist <= threshold else "rear"
                my_info.append(p_data)

            elif p.owner == player.opponent():
                enemy_planet += 1
                enemy_ships_on_planet += p.n_ships
                enemy_growth += p.growth_rate
                p_data["owner"] = "enemy"
                enemy_info.append(p_data)

            elif p.owner == Player.Neutral:
                neutral_planets += 1
                neutral_ships_on_planet += p.n_ships
                p_data["owner"] = "neutral"
                neutral_info.append(p_data)

        my_total_ships = my_ships_on_planet + my_ships_on_flight
        enemy_total_ships = enemy_ships_on_planet + enemy_ships_on_flight

        return json.dumps({
            "my_planet": my_planet,
            "my_total_ships": int(my_total_ships),
            "my_ships_on_flight": int(my_ships_on_flight),
            "my_growth": round(my_growth, 2),
            "my_info": my_info,

            "enemy_planet": enemy_planet,
            "enemy_total_ships": int(enemy_total_ships),
            "enemy_ships_on_flight": int(enemy_ships_on_flight),
            "enemy_growth": round(enemy_growth, 2),
            "enemy_info": enemy_info,

            "neutral_planets": neutral_planets,
            "neutral_ships_on_planet": int(neutral_ships_on_planet),
            "neutral_info": neutral_info,

            "game_tick": game_state.game_tick
        }, indent=2)
    
    
    def _get_llm_output(self, state_summary: str) -> str:
        prompt = self.prompt_template.replace("{state_summary}", state_summary)
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

    def _validate(self, text: str, my_planet_ids):
        validated_strategy = {}
        default_action = "N"
        valid_actions = {"A", "P", "E", "N"}

        try:
            clean_text = re.sub(r'```json|```', '', text).strip()
            data = json.loads(clean_text)

            for pl_id in my_planet_ids:
                pl_id_str = str(pl_id)
                if pl_id_str in data and data[pl_id_str] in valid_actions:
                    validated_strategy[pl_id] = data[pl_id_str]
                else:
                    print('change to default action')
                    validated_strategy[pl_id] = default_action
    
        except Exception as e:
            print(f"[Validation Error] {e}. Using default 'N' for all.")
            for pl_id in my_planet_ids:
                validated_strategy[pl_id] = default_action
                
        return validated_strategy
    
    def run(self, get_state_func, get_player_func):
        while self.running:
            current_time = time.time()
            state = get_state_func()
            my_player = get_player_func()
            
            if state and my_player:
                if current_time - self.last_update_time >= self.update_interval:
                    my_planet_ids = [p.id for p in state.planets if p.owner == my_player]
                    summary = self._parse_state(state, my_player)

                    print(f"--- Sending to LLM ---") 
                    start = time.perf_counter()
                    raw_response = self._get_llm_output(summary)
                    new_strategy = self._validate(raw_response, my_planet_ids)
                    finish = time.perf_counter() - start
                    print(f"Response time: {finish:.2f} seconds")

                    if new_strategy:
                        with self.lock:
                            self.current_strategy = new_strategy
                        
                        summary_data = json.loads(summary)
                        print(f"[LLM State] Me: {summary_data['my_planet']} planets | Enemy: {summary_data['enemy_planet']} planets")
                        print("--- Strategy Sample ---")
                        count = 0
                        for id, action in new_strategy.items():
                            print(f"  Planet {id:2}: [{action}]")
                            count += 1
                            if count >= 5:
                                break
                        if len(new_strategy) > 5:
                            print("  ...")
                        print("-----------------------\n")
                    else:
                        print(f"[LLM Update] Invalid output: {raw_response}\n")
                    
                    self.last_update_time = time.time()
            time.sleep(0.1)



'''ObserverAgent -- нужен для тестирования LLM'''
import os
from dotenv import load_dotenv
load_dotenv() 

class ObserverAgent(PlanetWarsPlayer):
    def __init__(self):
        super().__init__()

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API ключ OPENROUTER_API_KEY не найден в .env")
        
        self.llm_agent = LLMAgent(
            api_key=api_key, 
            base_url="https://openrouter.ai/api/v1"
        ) 
        self.latest_state = None
        
        self.llm_thread = threading.Thread(
            target=self.llm_agent.run,
            args=(
                lambda: self.latest_state,
                lambda: getattr(self, 'player', None)
            ),
            daemon=True
        )
        self.llm_thread.start()

    def get_action(self, game_state: GameState) -> Action:
        self.latest_state = game_state
        time.sleep(0.05) 
        return Action.do_nothing()
    
    def get_agent_type(self) -> str:
        return "LLM Observer"
    

