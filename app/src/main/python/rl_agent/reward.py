"""
Reward calculator for DQN training.

Computes per-tick reward based on changes in ship counts, planet ownership,
and **command compliance** (whether the agent followed the LLM's order).

Reward formula:
  reward = (Δmy_ships - Δenemy_ships) * 0.005
         + Δmy_planets * 5.0
         - Δenemy_planets * 5.0
         + command_bonus (±5.0 / −1.0)
         + 20.0  (if we won)
         - 20.0  (if we lost)
"""

from typing import Optional
from core.game_state import GameState, Player, Action


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
        action: Optional[Action] = None,
        llm_command: str = "",
    ) -> float:
        """
        Compute reward for the latest tick.

        Args:
            game_state: Game state AFTER the tick was executed.
            done: Whether the game ended this tick.
            winner: The winning player if done, else None.
            action: The Action that was taken (for command compliance).
            llm_command: The LLM command letter for the source planet
                         ('A', 'P', 'E', 'N', or '' if unavailable).

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

        # ── Command Compliance Bonus ──────────────────────────────────
        if llm_command and action is not None:
            reward += self._command_bonus(action, llm_command, game_state)

        # ── Win / Loss ────────────────────────────────────────────────
        if done and winner is not None:
            if winner == self.player:
                reward += 20.0
            elif winner == self.player.opponent():
                reward -= 20.0

        # Update state for next call
        self._prev_my_ships = my_ships
        self._prev_enemy_ships = enemy_ships
        self._prev_my_planets = my_planets
        self._prev_enemy_planets = enemy_planets

        return reward

    def _command_bonus(
        self, action: Action, llm_command: str, game_state: GameState
    ) -> float:
        """
        Reward the agent for following the LLM's order.

        +0.5 if the action matches the command.
        -0.1 if the action contradicts the command.
        """
        # DO_NOTHING action
        is_noop = (
            action.source_planet_id == -1
            or action.num_ships <= 0
        )

        if llm_command == "N":
            return 0.5 if is_noop else -0.1

        if is_noop:
            # LLM said do something, but agent did nothing
            return -0.1

        # Find the target planet to determine its owner
        target = None
        for p in game_state.planets:
            if p.id == action.destination_planet_id:
                target = p
                break

        if target is None:
            return 0.0

        if llm_command == "A" and target.owner == self.player.opponent():
            return 0.5
        elif llm_command == "P" and target.owner == self.player:
            return 0.5
        elif llm_command == "E" and target.owner == Player.Neutral:
            return 0.5
        else:
            return -0.1

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
