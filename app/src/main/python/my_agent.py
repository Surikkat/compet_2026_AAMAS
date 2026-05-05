from typing import Optional
from core.game_state import GameState, Action
from agents.planet_wars_agent import PlanetWarsPlayer
import random

class MyPythonAgent(PlanetWarsPlayer):
    """Мой первый Python-агент для Planet Wars"""

    def get_action(self, game_state: GameState) -> Action:
        # Найти свою планету с кораблями
        my_planets = [p for p in game_state.planets
                      if p.owner == self.player and p.n_ships > 1]
        if not my_planets:
            return Action.do_nothing()

        # Выбрать любую не свою планету
        targets = [p for p in game_state.planets
                   if p.owner != self.player]
        if not targets:
            return Action.do_nothing()

        src = random.choice(my_planets)
        dst = random.choice(targets)

        return Action(
            playerId=self.player,
            sourcePlanetId=src.id,
            destinationPlanetId=dst.id,
            numShips=int(src.n_ships // 2)
        )

    def get_agent_type(self) -> str:
        return "My Python Agent v0.1"