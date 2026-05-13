"""
Reward calculator for DQN training.

Computes per-tick reward based on changes in ship counts and planet ownership.

Reward formula:
  reward = (Δmy_ships - Δenemy_ships) * 0.01
         + Δmy_planets * 10.0
         - Δenemy_planets * 10.0
         + 100.0  (if we won)
         - 100.0  (if we lost)
"""

from typing import Optional
from core.game_state import GameState, Player


class RewardCalculator:
    """Calculates per-step rewards for RL training."""

    def __init__(self, player: Player):
        self.player = player
        self._prev_my_ships: float = 0.0
        self._prev_enemy_ships: float = 0.0
        self._prev_my_planets: int = 0
        self._prev_enemy_planets: int = 0

    def reset(self, game_state: GameState) -> None:
        """Initialize tracking from the starting game state."""
        self._prev_my_ships = self._total_ships(game_state, self.player)
        self._prev_enemy_ships = self._total_ships(
            game_state, self.player.opponent()
        )
        self._prev_my_planets = self._planet_count(game_state, self.player)
        self._prev_enemy_planets = self._planet_count(
            game_state, self.player.opponent()
        )

    def compute(
        self,
        game_state: GameState,
        done: bool,
        winner: Optional[Player],
    ) -> float:
        """
        Compute reward for the latest tick.

        Args:
            game_state: Game state AFTER the tick was executed.
            done: Whether the game ended this tick.
            winner: The winning player if done, else None.

        Returns:
            Float reward value.
        """
        my_ships = self._total_ships(game_state, self.player)
        enemy_ships = self._total_ships(game_state, self.player.opponent())
        my_planets = self._planet_count(game_state, self.player)
        enemy_planets = self._planet_count(game_state, self.player.opponent())

        # Deltas
        d_my_ships = my_ships - self._prev_my_ships
        d_enemy_ships = enemy_ships - self._prev_enemy_ships
        d_my_planets = my_planets - self._prev_my_planets
        d_enemy_planets = enemy_planets - self._prev_enemy_planets

        reward = (d_my_ships - d_enemy_ships) * 0.005
        reward += d_my_planets * 5.0
        reward -= d_enemy_planets * 5.0

        if done and winner is not None:
            if winner == self.player:
                reward += 50.0
            elif winner == self.player.opponent():
                reward -= 50.0

        # Update state for next call
        self._prev_my_ships = my_ships
        self._prev_enemy_ships = enemy_ships
        self._prev_my_planets = my_planets
        self._prev_enemy_planets = enemy_planets

        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_ships(game_state: GameState, player: Player) -> float:
        ships = sum(p.n_ships for p in game_state.planets if p.owner == player)
        # Also count ships in transit (transporters)
        for p in game_state.planets:
            if p.transporter is not None and p.transporter.owner == player:
                ships += p.transporter.n_ships
        return ships

    @staticmethod
    def _planet_count(game_state: GameState, player: Player) -> int:
        return sum(1 for p in game_state.planets if p.owner == player)
