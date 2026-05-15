import time
import json
import re
from typing import Optional
from openai import OpenAI
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

class PureLLMAgent(PlanetWarsPlayer):
    """LLM-only: запрашиваем действие и выполняем."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._calls = 0
        self._success = 0
        self._fallback = 0
        
        self.prompt = """Planet Wars RTS. You are {player}.

Current state of ALL planets:
{planets}

DECIDE: Send ships from ONE of your planets to ONE target.
Return JSON: {{"source": YOUR_PLANET_ID, "target": TARGET_PLANET_ID, "ships": NUMBER}}

CRITICAL RULES:
1. Only send from planets you OWN
2. Target must be enemy or neutral planet (NOT your own)
3. Leave at least 1 ship on source planet for defense
4. Combat is 1:1 (your ships - enemy ships = remaining)
5. Neutrals: easy to capture, capture them early
6. Enemy: attack when you have more ships than they do
7. If enemy is stronger, focus on capturing neutrals to grow

Output ONLY the JSON. No explanation."""
    
    def prepare_to_play_as(
        self, player: Player, params: GameParams, opponent: Optional[str] = None
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        self._calls += 1
        
        # Формируем полное описание планет
        planets_desc = []
        for p in game_state.planets:
            owner_str = "YOU" if p.owner == self.player else ("ENEMY" if p.owner == self.player.opponent() else "NEUTRAL")
            planets_desc.append(
                f"Planet {p.id}: owner={owner_str}, ships={int(p.n_ships)}, "
                f"growth={p.growth_rate:.2f}, pos=({p.position.x:.0f},{p.position.y:.0f})"
            )
        
        prompt = self.prompt.format(
            player=str(self.player),
            planets="\n".join(planets_desc)
        )
        
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
                timeout=2.0
            )
            
            text = response.choices[0].message.content
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found")
            
            data = json.loads(match.group(0))
            src_id = int(data["source"])
            tgt_id = int(data["target"])
            ships = int(data.get("ships", 0))
            
            # Валидация
            source = next((p for p in game_state.planets if p.id == src_id), None)
            if not source or source.owner != self.player:
                raise ValueError(f"Invalid source {src_id}")
            if source.n_ships < 2:
                raise ValueError(f"Not enough ships on source {src_id}")
            
            if ships <= 0:
                ships = max(1, int(source.n_ships * 0.5))
            ships = min(ships, source.n_ships - 1)
            
            self._success += 1
            return Action(
                playerId=self.player,
                sourcePlanetId=src_id,
                destinationPlanetId=tgt_id,
                numShips=ships
            )
        except Exception as e:
            self._fallback += 1
            if self._success == 0:
                print(f"[LLM #{self._calls}] {e}")
            return self._fallback_action(game_state)
    
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
        return "Pure LLM Direct v6.0"
    
    def get_stats(self) -> dict:
        return {
            "calls": self._calls,
            "llm_success": self._success,
            "fallback": self._fallback
        }

