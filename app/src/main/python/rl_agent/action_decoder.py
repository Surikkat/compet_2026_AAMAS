"""
Action decoder for Pairwise DQN agent.

Converts a discrete action index (0–4) and a (source, target) pair
into a concrete Action object that the game engine can execute.

Action indices:
  0 → DO_NOTHING  (send 0% ships)
  1 → send 25% of source ships to target
  2 → send 50% of source ships to target
  3 → send 75% of source ships to target
  4 → send ~100% of source ships to target (keep 1)
"""

import math
from core.game_state import Action, Planet, Player, GameState


# Fraction of ships to send for each action index
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
NUM_ACTIONS = len(FRACTIONS)


class ActionDecoder:
    """Decodes DQN action indices into game Actions."""

    def decode(
        self,
        action_idx: int,
        source: Planet,
        target: Planet,
        player: Player,
    ) -> Action:
        """
        Convert an action index + planet pair into a game Action.

        Args:
            action_idx: Integer 0–4 representing the fraction of ships to send.
            source: The planet we're sending ships FROM.
            target: The planet we're sending ships TO.
            player: Our player identity.

        Returns:
            An Action object. Returns DO_NOTHING if action_idx == 0
            or if the source can't send ships.
        """
        if action_idx == 0:
            return Action.do_nothing()

        if source.transporter is not None:
            return Action.do_nothing()

        fraction = FRACTIONS[action_idx]
        num_ships = source.n_ships * fraction

        # For 100%, keep at least 1 ship on the planet
        if action_idx == 4:
            num_ships = max(source.n_ships - 1, 0)

        # Need to send at least 1 ship
        num_ships = math.floor(num_ships)
        if num_ships < 1:
            return Action.do_nothing()

        return Action(
            player_id=player,
            source_planet_id=source.id,
            destination_planet_id=target.id,
            num_ships=float(num_ships),
        )