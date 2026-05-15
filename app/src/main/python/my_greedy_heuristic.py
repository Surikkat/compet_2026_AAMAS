from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams


class MyGreedyHeuristicAgent(PlanetWarsPlayer):
    """Greedy v2.0 — умный захват нейтралов с низким порогом."""
    
    def get_action(self, game_state: GameState) -> Action:
        my_planets = [p for p in game_state.planets if p.owner == self.player and p.transporter is None]
        neutral_planets = [p for p in game_state.planets if p.owner == Player.Neutral]
        enemy_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]
        
        if not my_planets:
            return Action.do_nothing()
        
        # Раздельная логика: нейтралы vs враги
        neutral_ready = [p for p in my_planets if p.n_ships > 2]  # Низкий порог для нейтралов
        enemy_ready = [p for p in my_planets if p.n_ships > 10]   # Высокий порог для врагов
        
        # Приоритет: захват нейтралов (если есть готовые планеты)
        if neutral_planets and neutral_ready:
            source = max(neutral_ready, key=lambda p: p.n_ships)
            
            # Heuristic для нейтралов: предпочитаем слабые, близкие, с высоким ростом
            def neutral_score(target):
                distance = source.position.distance(target.position)
                return target.n_ships + distance * 0.5 - 3 * target.growth_rate
            
            target = min(neutral_planets, key=neutral_score)
            
            # Сколько нужно для захвата
            ships_needed = int(target.n_ships) + 1
            
            if source.n_ships > ships_needed:
                # Отправляем сколько нужно + 20% запас, но не больше 80%
                ships_to_send = min(int(ships_needed * 1.2) + 1, int(source.n_ships * 0.8))
                return Action(
                    player_id=self.player,
                    source_planet_id=source.id,
                    destination_planet_id=target.id,
                    num_ships=ships_to_send
                )
        
        # Атака врагов (оригинальная логика)
        if enemy_planets and enemy_ready:
            source = max(enemy_ready, key=lambda p: p.n_ships)
            
            def target_score(target):
                distance = source.position.distance(target.position)
                ship_strength = target.n_ships * 1.5  # Враги опаснее
                return ship_strength + distance - 2 * target.growth_rate
            
            target = min(enemy_planets, key=target_score)
            
            distance = source.position.distance(target.position)
            eta = distance / self.params.transporter_speed
            estimated_defense = target.n_ships + target.growth_rate * eta
            
            if source.n_ships > estimated_defense:
                return Action(
                    player_id=self.player,
                    source_planet_id=source.id,
                    destination_planet_id=target.id,
                    num_ships=source.n_ships / 2
                )
        
        return Action.do_nothing()
    
    def get_agent_type(self) -> str:
        return "My Greedy Heuristic v3.0"