"""
State encoder for Pairwise DQN agent.

Encodes (source_planet, target_planet) pairs into fixed-size feature vectors
that include per-planet features, pair-specific features, global context,
and LLM strategy vectors.

Feature vector layout (23 dimensions):
  Source planet (6):  n_ships, growth_rate, dist_nearest_enemy,
                      dist_nearest_ally, dist_nearest_neutral, has_transporter
  Target planet (7):  owner_relation, n_ships, growth_rate, distance_to_source,
                      incoming_balance, dist_nearest_enemy, dist_nearest_ally
  Global context (6): ship_ratio, enemy_ships_norm, my_planet_frac,
                       enemy_planet_frac, neutral_planet_frac, game_progress
  LLM strategy (4):   p_accumulate, p_attack_enemy, p_transfer_ally, p_attack_neutral
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional

from core.game_state import GameState, GameParams, Planet, Player


# Default LLM strategy (uniform) used as fallback
DEFAULT_LLM_VEC = [0.25, 0.25, 0.25, 0.25]

# Feature dimension
FEATURE_DIM = 25


class StateEncoder:
    """Encodes game state into feature vectors for DQN pairwise evaluation."""

    def __init__(self, params: GameParams):
        self.params = params
        self.diag = math.sqrt(params.width ** 2 + params.height ** 2)
        self.max_growth = max(params.max_growth_rate, 1e-6)
        self.ship_norm = 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_all_pairs(
        self,
        game_state: GameState,
        player: Player,
        llm_strategy: Dict[int, List[float]],
    ) -> Tuple[np.ndarray, List[Tuple[Planet, Planet]]]:
        """
        Encode all valid (source, target) pairs into a batched feature matrix.

        Args:
            game_state: Current game state.
            player: Our player identity.
            llm_strategy: Mapping planet_id -> [4 floats] from the LLM advisor.

        Returns:
            features: np.ndarray of shape (N_pairs, 23), dtype float32.
            pair_info: list of (source_planet, target_planet) tuples aligned
                       with the rows of `features`.
        """
        planets = game_state.planets
        n = len(planets)

        # Valid sources: owned by us, transporter free, has ships to send
        sources = [
            p for p in planets
            if p.owner == player and p.transporter is None and p.n_ships > 1
        ]
        if not sources:
            return np.empty((0, FEATURE_DIM), dtype=np.float32), []

        # --- Precompute shared data -------------------------------------------
        positions = np.array(
            [[p.position.x, p.position.y] for p in planets], dtype=np.float64
        )

        # Full pairwise distance matrix (n x n)
        diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))         # (n, n)

        # Planet category masks
        enemy_mask = np.array([p.owner == player.opponent() for p in planets])
        ally_mask = np.array([p.owner == player for p in planets])
        neutral_mask = np.array([p.owner == Player.Neutral for p in planets])

        # Nearest enemy/ally/neutral for every planet
        nearest_enemy = self._nearest_dist(dist_matrix, enemy_mask)
        nearest_ally_excl = self._nearest_dist_exclude_self(dist_matrix, ally_mask)
        nearest_neutral = self._nearest_dist(dist_matrix, neutral_mask)
        avg_enemy = self._avg_dist(dist_matrix, enemy_mask)

        # Incoming ship balance per planet
        incoming = self._compute_incoming(game_state, player)

        # Global features (same for all pairs this tick)
        global_feats = self._global_features(game_state, player)

        # --- Build pair feature matrix ----------------------------------------
        features: List[np.ndarray] = []
        pair_info: List[Tuple[Planet, Planet]] = []

        for src in sources:
            sid = src.id
            src_feats = np.array([
                src.n_ships / self.ship_norm,
                src.growth_rate / self.max_growth,
                nearest_enemy[sid] / self.diag,
                nearest_ally_excl[sid] / self.diag,
                nearest_neutral[sid] / self.diag,
                avg_enemy[sid] / self.diag,
                0.0,  # has_transporter is always False here (filtered above)
            ], dtype=np.float32)

            llm_vec = np.array(
                llm_strategy.get(src.id, DEFAULT_LLM_VEC), dtype=np.float32
            )

            for tgt in planets:
                if tgt.id == sid:
                    continue  # skip self-targeting

                tid = tgt.id
                # Owner relation: ally=1, neutral=0, enemy=-1
                if tgt.owner == player:
                    owner_rel = 1.0
                elif tgt.owner == Player.Neutral:
                    owner_rel = 0.0
                else:
                    owner_rel = -1.0

                inc = incoming.get(tid, (0.0, 0.0))
                inc_balance = (inc[0] - inc[1]) / self.ship_norm

                tgt_feats = np.array([
                    owner_rel,
                    tgt.n_ships / self.ship_norm,
                    tgt.growth_rate / self.max_growth,
                    dist_matrix[sid, tid] / self.diag,
                    inc_balance,
                    nearest_enemy[tid] / self.diag,
                    nearest_ally_excl[tid] / self.diag,
                    avg_enemy[tid] / self.diag,
                ], dtype=np.float32)

                pair_feat = np.concatenate([src_feats, tgt_feats, global_feats, llm_vec])
                features.append(pair_feat)
                pair_info.append((src, tgt))

        if not features:
            return np.empty((0, FEATURE_DIM), dtype=np.float32), []

        return np.stack(features).astype(np.float32), pair_info

    def encode_pair(
        self,
        source: Planet,
        target: Planet,
        game_state: GameState,
        player: Player,
        llm_strategy: Dict[int, List[float]],
    ) -> np.ndarray:
        """Encode a single (source, target) pair. Shape (23,)."""
        # Reuses encode_all_pairs with a filter — simple but not the fastest.
        # Used mainly for storing next-state in replay buffer.
        planets = game_state.planets
        n = len(planets)

        positions = np.array(
            [[p.position.x, p.position.y] for p in planets], dtype=np.float64
        )
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        enemy_mask = np.array([p.owner == player.opponent() for p in planets])
        ally_mask = np.array([p.owner == player for p in planets])
        neutral_mask = np.array([p.owner == Player.Neutral for p in planets])

        nearest_enemy = self._nearest_dist(dist_matrix, enemy_mask)
        nearest_ally_excl = self._nearest_dist_exclude_self(dist_matrix, ally_mask)
        nearest_neutral = self._nearest_dist(dist_matrix, neutral_mask)
        avg_enemy = self._avg_dist(dist_matrix, enemy_mask)

        incoming = self._compute_incoming(game_state, player)
        global_feats = self._global_features(game_state, player)

        sid = source.id
        tid = target.id

        src_feats = np.array([
            source.n_ships / self.ship_norm,
            source.growth_rate / self.max_growth,
            nearest_enemy[sid] / self.diag,
            nearest_ally_excl[sid] / self.diag,
            nearest_neutral[sid] / self.diag,
            avg_enemy[sid] / self.diag,
            1.0 if source.transporter is not None else 0.0,
        ], dtype=np.float32)

        if target.owner == player:
            owner_rel = 1.0
        elif target.owner == Player.Neutral:
            owner_rel = 0.0
        else:
            owner_rel = -1.0

        inc = incoming.get(tid, (0.0, 0.0))
        inc_balance = (inc[0] - inc[1]) / self.ship_norm

        tgt_feats = np.array([
            owner_rel,
            target.n_ships / self.ship_norm,
            target.growth_rate / self.max_growth,
            dist_matrix[sid, tid] / self.diag,
            inc_balance,
            nearest_enemy[tid] / self.diag,
            nearest_ally_excl[tid] / self.diag,
            avg_enemy[tid] / self.diag,
        ], dtype=np.float32)

        llm_vec = np.array(
            llm_strategy.get(source.id, DEFAULT_LLM_VEC), dtype=np.float32
        )

        return np.concatenate([src_feats, tgt_feats, global_feats, llm_vec])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _global_features(self, game_state: GameState, player: Player) -> np.ndarray:
        """Compute 6-dim global feature vector."""
        planets = game_state.planets
        total = len(planets)

        my_ships = sum(p.n_ships for p in planets if p.owner == player)
        enemy_ships = sum(p.n_ships for p in planets if p.owner == player.opponent())
        my_count = sum(1 for p in planets if p.owner == player)
        enemy_count = sum(1 for p in planets if p.owner == player.opponent())
        neutral_count = sum(1 for p in planets if p.owner == Player.Neutral)

        total_ships = my_ships + enemy_ships + 1e-6  # avoid div by zero

        return np.array([
            my_ships / total_ships,
            enemy_ships / 1000.0,
            my_count / total,
            enemy_count / total,
            neutral_count / total,
            game_state.game_tick / max(self.params.max_ticks, 1),
        ], dtype=np.float32)

    def _compute_incoming(
        self, game_state: GameState, player: Player
    ) -> Dict[int, Tuple[float, float]]:
        """
        For each planet, compute (allied_incoming, enemy_incoming) ship counts
        from active transporters heading there.
        """
        incoming: Dict[int, List[float]] = {}
        for planet in game_state.planets:
            if planet.transporter is not None:
                t = planet.transporter
                dest = t.destination_index
                if dest not in incoming:
                    incoming[dest] = [0.0, 0.0]
                if t.owner == player:
                    incoming[dest][0] += t.n_ships
                else:
                    incoming[dest][1] += t.n_ships
        return {k: (v[0], v[1]) for k, v in incoming.items()}

    @staticmethod
    def _nearest_dist(dist_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        For each planet (row), find the minimum distance to any planet
        satisfying `mask`. Returns array of length n.
        """
        n = dist_matrix.shape[0]
        if not mask.any():
            return np.full(n, 1e6)
        sub = dist_matrix[:, mask]
        return sub.min(axis=1)

    @staticmethod
    def _nearest_dist_exclude_self(
        dist_matrix: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Like _nearest_dist, but excludes self-distance (diagonal = 0).
        Useful for allies where the planet itself is in the mask.
        """
        n = dist_matrix.shape[0]
        if not mask.any():
            return np.full(n, 1e6)
        # Set diagonal to inf so self is never the nearest
        dm = dist_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        sub = dm[:, mask]
        if sub.size == 0:
            return np.full(n, 1e6)
        return sub.min(axis=1)

    @staticmethod
    def _avg_dist(dist_matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        For each planet, find the average distance to all planets satisfying mask.
        """
        n = dist_matrix.shape[0]
        if not mask.any():
            return np.full(n, 1e6)
        sub = dist_matrix[:, mask]
        return sub.mean(axis=1)
