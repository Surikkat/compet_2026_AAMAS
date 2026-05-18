import math
from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams


class MyGreedyHeuristicAgent(PlanetWarsPlayer):
    """Greedy v3.0 — умный захват нейтралов с низким порогом (улучшенный)."""
    
    def get_action(self, game_state: GameState) -> Action:
        my_planets = [p for p in game_state.planets if p.owner == self.player]
        if not my_planets:
            return Action.do_nothing()

        incoming_my = {}
        incoming_enemy = {}
        our_targets = set()
        for p in game_state.planets:
            t = p.transporter
            if t:
                dest_id = t.destination_index
                if t.owner == self.player:
                    incoming_my[dest_id] = incoming_my.get(dest_id, 0.0) + t.n_ships
                    our_targets.add(dest_id)
                elif t.owner == self.player.opponent():
                    incoming_enemy[dest_id] = incoming_enemy.get(dest_id, 0.0) + t.n_ships

        free_planets = [p for p in my_planets if p.transporter is None]
        neutral_planets = [p for p in game_state.planets if p.owner == Player.Neutral]
        enemy_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]

        # defense
        worst_deficit = 0.0
        threatened = None
        for p in my_planets:
            enemy_incoming = incoming_enemy.get(p.id, 0.0)
            if enemy_incoming > 0:
                net = p.n_ships + incoming_my.get(p.id, 0.0) - enemy_incoming
                if net < 0 and abs(net) > worst_deficit:
                    worst_deficit = abs(net)
                    threatened = p

        if threatened:
            ships_needed = worst_deficit * 1.1 + 1
            best_helper = None
            best_dist = float('inf')
            for src in free_planets:
                if src.id != threatened.id and src.n_ships - 2 >= ships_needed:
                    d = src.position.distance(threatened.position)
                    if d < best_dist:
                        best_dist = d
                        best_helper = src
            if best_helper:
                ships_to_send = min(ships_needed, best_helper.n_ships * 0.85)
                return Action(
                    player_id=self.player,
                    source_planet_id=best_helper.id,
                    destination_planet_id=threatened.id,
                    num_ships=ships_to_send
                )

        # unified expand + attack
        best_score = -1.0
        best_action = None
        
        for target in game_state.planets:
            if target.owner == self.player:
                continue
            if target.id in our_targets:
                continue
                
            is_neutral = target.owner == Player.Neutral
            
            for source in free_planets:
                if is_neutral and source.n_ships <= 2:
                    continue
                if not is_neutral and source.n_ships <= 5:
                    continue
                    
                dist = source.position.distance(target.position)
                if dist < 1e-6: dist = 1e-6
                eta = dist / self.params.transporter_speed
                
                if is_neutral:
                    ships_raw = target.n_ships + 1 - incoming_my.get(target.id, 0.0)
                    if ships_raw <= 0:
                        continue
                    ships_needed = ships_raw + max(1, ships_raw * 0.15)
                    growth_weight = 4.0
                else:
                    defense = target.n_ships + target.growth_rate * eta - incoming_my.get(target.id, 0.0) + incoming_enemy.get(target.id, 0.0)
                    if defense <= 0:
                        continue
                    ships_needed = defense + max(1, defense * 0.15)
                    growth_weight = 2.5
                    
                if source.n_ships <= ships_needed:
                    continue
                    
                score = target.growth_rate * growth_weight / (ships_needed * dist)
                if score > best_score:
                    best_score = score
                    ships_to_send = min(ships_needed, source.n_ships * 0.85)
                    best_action = Action(
                        player_id=self.player,
                        source_planet_id=source.id,
                        destination_planet_id=target.id,
                        num_ships=ships_to_send
                    )
        
        if best_action:
            return best_action

        # fallback reinforce
        if len(free_planets) >= 2 and enemy_planets:
            ex = sum(p.position.x for p in enemy_planets) / len(enemy_planets)
            ey = sum(p.position.y for p in enemy_planets) / len(enemy_planets)
            best_rear = None
            best_front = None
            max_rear_d = -1.0
            min_front_d = float('inf')
            for p in free_planets:
                d = math.hypot(p.position.x - ex, p.position.y - ey)
                if p.n_ships > 8 and d > max_rear_d:
                    max_rear_d = d
                    best_rear = p
                if d < min_front_d:
                    min_front_d = d
                    best_front = p
            if best_rear and best_front and best_rear.id != best_front.id:
                ships_to_send = min(best_rear.n_ships * 0.5, best_rear.n_ships - 4)
                if ships_to_send > 2:
                    return Action(
                        player_id=self.player,
                        source_planet_id=best_rear.id,
                        destination_planet_id=best_front.id,
                        num_ships=ships_to_send
                    )

        return Action.do_nothing()
        
    def get_agent_type(self) -> str:
        return "My Greedy Heuristic v3.0"