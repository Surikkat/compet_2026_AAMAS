from typing import Optional
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer
import os
from my_agent_llm import PureLLMAgent

class MyPythonAgent(PlanetWarsPlayer):
    """Fallback-only агент с опциональным LLM."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, use_llm: bool = False):
        super().__init__()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if use_llm and not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        self.llm_agent = PureLLMAgent(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            use_llm=use_llm
        )
    
    def prepare_to_play_as(
        self, player: Player, params: GameParams, opponent: Optional[str] = None
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        self.llm_agent.prepare_to_play_as(player, params, opponent)
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        return self.llm_agent.get_action(game_state)
    
    def get_agent_type(self) -> str:
        return "Pure LLM Agent v4.0"