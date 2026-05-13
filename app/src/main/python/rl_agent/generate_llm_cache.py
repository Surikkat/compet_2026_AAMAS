"""
Script to generate an offline LLM strategy cache for V2 (Per-planet).
Runs headless games and queries GPT-4o-mini for strategies, saving them to JSON.
"""

import os
import sys
import json
import time
import random
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add parent dir to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.game_state import GameState, GameParams, Player
from core.game_state_factory import GameStateFactory
from core.forward_model import ForwardModel
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.random_agents import CarefulRandomAgent

from v2_my_agent_llm import LLMAgent as V2LLMAgent


def extract_planet_state_vector(planet, state: GameState, player: Player) -> list:
    """Extract 3-dimensional continuous state vector for a single planet."""
    enemy_planets = [p for p in state.planets if p.owner == player.opponent()]
    if not enemy_planets:
        return [0.0, 0.0, 0.0]
        
    avg_enemy_x = sum(p.position.x for p in enemy_planets) / len(enemy_planets)
    avg_enemy_y = sum(p.position.y for p in enemy_planets) / len(enemy_planets)
    
    my_planets = [p for p in state.planets if p.owner == player]
    if not my_planets:
        return [0.0, 0.0, 0.0]
        
    distances = [((p.position.x - avg_enemy_x)**2 + (p.position.y - avg_enemy_y)**2)**0.5 for p in my_planets]
    threshold = sum(distances) / len(distances)
    
    dist = ((planet.position.x - avg_enemy_x)**2 + (planet.position.y - avg_enemy_y)**2)**0.5
    is_frontline = 1.0 if dist <= threshold else 0.0
    
    my_ships = sum(p.n_ships for p in state.planets if p.owner == player) + 1e-5
    ship_ratio = planet.n_ships / my_ships
    
    max_growth = max(p.growth_rate for p in state.planets) + 1e-5
    growth_ratio = planet.growth_rate / max_growth
    
    return [is_frontline, ship_ratio, growth_ratio]


def generate_cache(num_queries: int = 50):
    """
    Generate dataset. For V2, one query produces strategies for all owned planets.
    50 queries * ~10 planets = 500 training examples for the dataset.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return
        
    print("Initializing GPT-4o-mini for V2 dataset generation...")
    # Using the LLMAgent but calling openai manually
    llm_agent = V2LLMAgent(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    cache_data = []
    
    # Run games to collect data
    queries_collected = 0
    pbar = tqdm(total=num_queries, desc="Generating Cache Queries")
    
    while queries_collected < num_queries:
        params = GameParams(
            num_planets=random.randint(10, 30),
            max_ticks=500
        )
        state = GameStateFactory(params).create_game()
        fm = ForwardModel(state, params)
        
        agent1 = CarefulRandomAgent()
        agent2 = GreedyHeuristicAgent()
        agent1.prepare_to_play_as(Player.Player1, params)
        agent2.prepare_to_play_as(Player.Player2, params)
        
        # Determine at what tick we want to query the LLM (random point in the game)
        query_tick = random.randint(20, 200)
        
        while not fm.is_terminal():
            if fm.state.game_tick == query_tick:
                # Time to query!
                my_planet_ids = [p.id for p in fm.state.planets if p.owner == Player.Player1]
                if not my_planet_ids:
                    break
                    
                state_summary = llm_agent._parse_state(fm.state, Player.Player1)
                # LLMAgent uses the replace method for prompt formatting
                prompt = llm_agent.prompt_template.replace("{state_summary}", state_summary)
                
                try:
                    response = llm_agent.client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=200,
                    )
                    raw_response = response.choices[0].message.content
                    strategy_dict = llm_agent._validate(raw_response, my_planet_ids)
                    
                    if strategy_dict:
                        # For each planet, store its features and the LLM's letter response
                        for pid, command in strategy_dict.items():
                            # Validate the planet is actually ours
                            planet = next((p for p in fm.state.planets if p.id == pid), None)
                            if planet and planet.owner == Player.Player1:
                                state_vec = extract_planet_state_vector(planet, fm.state, Player.Player1)
                                cache_data.append({
                                    "state_vector": state_vec,
                                    "strategy": command
                                })
                        queries_collected += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"\\nAPI Error: {e}")
                    time.sleep(2)
                    
                break # Move to next game after 1 query
                
            actions = {
                Player.Player1: agent1.get_action(fm.state.model_copy(deep=True)),
                Player.Player2: agent2.get_action(fm.state.model_copy(deep=True))
            }
            fm.step(actions)
            
    pbar.close()
    
    output_path = os.path.join(os.path.dirname(__file__), "llm_cache.json")
    with open(output_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"Successfully generated {len(cache_data)} planet examples and saved to {output_path}")


if __name__ == "__main__":
    generate_cache(50)
