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
        self.current_strategy = [0.0, 0.0, 0.0, 1.0] # Nothing to do in base strategy

        self.last_update_time = 0.0
        self.update_interval = 5.0 # seconds
        self.update_interval = 4.0
        self.running = True
        self.lock = threading.Lock()

        self.prompt_path = "/home/maria/phystech/projects/game_planet_wars/compet_2026_AAMAS/app/src/main/python/prompt_en.md"

        # Load prompt template
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print("[Warning] prompt.md not found! Using default template.")
            self.prompt_template = "State:\n{state_summary}\nOutput [A, P, E, N] as floats summing to 1.0."
    
    def _parse_state(self, game_state: GameState, player: Player) -> str:
        my_planet = 0
        my_total_ships, my_ships_on_planet, my_ships_on_flight = 0.0, 0.0, 0.0
        my_growth = 0.0 

        enemy_planet = 0
        enemy_total_ships, enemy_ships_on_planet, enemy_ships_on_flight = 0.0, 0.0, 0.0
        enemy_growth = 0.0

        neutral_planets = 0
        neutral_ships_on_planet=0.0

        for p in game_state.planets:
            if p.owner == player:
                my_planet += 1
                my_ships_on_planet += p.n_ships
                my_growth += p.growth_rate

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
            "my_planet": my_planet,
            "my_total_ships": int(my_total_ships),
            "my_ships_on_flight": int(my_ships_on_flight),
            "my_growth": round(my_growth, 2),
            "enemy_planet": enemy_planet,
            "enemy_total_ships": int(enemy_total_ships),
            "enemy_ships_on_flight": int(enemy_ships_on_flight),
            "enemy_growth": round(enemy_growth, 2),
            "neutral_planets": neutral_planets,
            "neutral_ships_on_planet": int(neutral_ships_on_planet),
        }, indent=2)
    

    def _get_llm_output(self, state_summary: str) -> str:
        prompt = self.prompt_template.format(state_summary=state_summary)
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

    def _validate(self, text: str) -> Optional[List[float]]:
            try:
                match = re.search(r'\[(.*?)\]', text)
                if not match: return None
                arr = json.loads(f"[{match.group(1)}]")
                if len(arr) == 4 and all(isinstance(x, (int, float)) for x in arr):
                    # Normalize
                    total = sum(arr)
                    if total > 0:
                        return [x / total for x in arr]
            except Exception:
                pass
            return None
    
    def run(self, get_state_func, get_player_func):
        while self.running:
            current_time = time.time()
            state = get_state_func()
            my_player = get_player_func()
            
            if state and my_player:
                if current_time - self.last_update_time >= self.update_interval:
                    summary = self._parse_state(state, my_player)
                    print(f"--- Sending to LLM ---") 
                    raw_response = self._get_llm_output(summary)
                    valid_vector = self._validate(raw_response)
                    
                    if valid_vector:
                        with self.lock:
                            self.current_strategy = valid_vector
                        print(f"[LLM State] Me: {json.loads(summary)['my_planet']} planets | Enemy: {json.loads(summary)['enemy_planet']} planets")
                        print(f"[LLM Update] Strategy [A, P, E, N]: {valid_vector}\n")
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
    

