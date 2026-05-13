"""
Script to generate an offline LLM strategy cache for V1 (Global).
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

from my_agent_llm import LLMAgent


def extract_state_vector(state: GameState, player: Player) -> list:
    """Extract 4-dimensional continuous state vector."""
    my_planets = sum(1 for p in state.planets if p.owner == player)
    enemy_planets = sum(1 for p in state.planets if p.owner == player.opponent())
    total_planets = len(state.planets)
    
    my_ships = sum(p.n_ships for p in state.planets if p.owner == player)
    enemy_ships = sum(p.n_ships for p in state.planets if p.owner == player.opponent())
    
    # Include ships in transit
    for p in state.planets:
        if p.transporter is not None:
            if p.transporter.owner == player:
                my_ships += p.transporter.n_ships
            elif p.transporter.owner == player.opponent():
                enemy_ships += p.transporter.n_ships
                
    total_ships = sum(p.n_ships for p in state.planets) + \
                  sum(p.transporter.n_ships for p in state.planets if p.transporter) + 1e-5
                  
    return [
        my_planets / total_planets,
        my_ships / total_ships,
        enemy_planets / total_planets,
        enemy_ships / total_ships
    ]


def generate_cache(num_examples: int = 150):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return
        
    print("Initializing GPT-4o-mini for dataset generation...")
    # Using the LLMAgent but we will manually call OpenAI to override the model
    llm_agent = LLMAgent(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    cache_data = []
    
    # Run games to collect data
    examples_collected = 0
    pbar = tqdm(total=num_examples, desc="Generating Cache")
    
    while examples_collected < num_examples:
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
                state_vec = extract_state_vector(fm.state, Player.Player1)
                state_summary = llm_agent._parse_state(fm.state, Player.Player1)
                prompt = llm_agent.prompt_template.format(state_summary=state_summary)
                
                try:
                    response = llm_agent.client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=30,
                    )
                    raw_response = response.choices[0].message.content
                    strategy = llm_agent._validate(raw_response)
                    
                    if strategy:
                        cache_data.append({
                            "state_vector": state_vec,
                            "strategy": strategy
                        })
                        examples_collected += 1
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
    print(f"Successfully generated {len(cache_data)} examples and saved to {output_path}")


if __name__ == "__main__":
    generate_cache(150)
