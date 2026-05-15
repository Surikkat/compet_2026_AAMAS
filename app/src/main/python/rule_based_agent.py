from typing import Optional, List
import math
from core.game_state import GameState, GameParams, Player, Action
from agents.planet_wars_agent import PlanetWarsPlayer

class RuleBasedAgent(PlanetWarsPlayer):
    """Агент на правилах логики."""
    
    def __init__(self):
        super().__init__()
        self._action_count = 0
    
    def prepare_to_play_as(
        self, player: Player, params: GameParams, opponent: Optional[str] = None
    ) -> str:
        super().prepare_to_play_as(player, params, opponent)
        return self.get_agent_type()
    
    def get_action(self, game_state: GameState) -> Action:
        self._action_count += 1
        
        my_planets = [p for p in game_state.planets if p.owner == self.player]
        enemy_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]
        neutral_planets = [p for p in game_state.planets if p.owner == Player.Neutral]
        
        if not my_planets:
            return Action.do_nothing()
        
        # Правило 1: Захватываем нейтральные планеты (ранняя игра)
        if neutral_planets:
            action = self._capture_neutral(my_planets, neutral_planets)
            if action:
                return action
        
        # Правило 2: Защищаем свои планеты от атак
        action = self._defend(my_planets, enemy_planets)
        if action:
            return action
        
        # Правило 3: Атакуем слабые планеты врага
        if enemy_planets:
            action = self._attack_enemy(my_planets, enemy_planets)
            if action:
                return action
        
        # Правило 4: Перебрасываем корабли между своими планетами
        action = self._reinforce(my_planets)
        if action:
            return action
        
        return Action.do_nothing()
    
    def _capture_neutral(self, my_planets: List, neutrals: List) -> Optional[Action]:
        """Захватить самую выгодную нейтральную планету."""
        # Сортируем нейтралов по привлекательности (growth rate / ships needed)
        best_target = None
        best_score = -1
        best_source = None
        best_ships = 0
        
        for neutral in neutrals:
            # Сколько кораблей нужно для захвата
            ships_needed = int(neutral.n_ships) + 1
            
            for src in my_planets:
                if src.n_ships > ships_needed + 1:  # +1 чтобы оставить защиту
                    # "Выгодность" = growth_rate / (ships_needed * distance)
                    distance = self._distance(src, neutral)
                    score = neutral.growth_rate / (ships_needed * max(1, distance))
                    
                    if score > best_score:
                        best_score = score
                        best_target = neutral
                        best_source = src
                        best_ships = ships_needed + max(1, int(ships_needed * 0.2))  # +20% запас
        
        if best_target and best_source:
            return Action(
                playerId=self.player,
                sourcePlanetId=best_source.id,
                destinationPlanetId=best_target.id,
                numShips=min(best_ships, best_source.n_ships - 1)
            )
        
        return None
    
    def _defend(self, my_planets: List, enemies: List) -> Optional[Action]:
        """Защитить планету, на которую летит вражеский флот."""
        for planet in my_planets:
            if planet.transporter and planet.transporter.owner != self.player:
                incoming_enemy = int(planet.transporter.n_ships)
                if planet.n_ships < incoming_enemy:
                    # Нужна помощь! Ищем ближайшую планету с кораблями
                    help_needed = incoming_enemy - int(planet.n_ships) + 1
                    
                    best_helper = None
                    best_distance = float('inf')
                    
                    for helper in my_planets:
                        if helper.id != planet.id and helper.n_ships > help_needed + 1:
                            distance = self._distance(helper, planet)
                            if distance < best_distance:
                                best_distance = distance
                                best_helper = helper
                    
                    if best_helper:
                        return Action(
                            playerId=self.player,
                            sourcePlanetId=best_helper.id,
                            destinationPlanetId=planet.id,
                            numShips=min(help_needed, best_helper.n_ships - 1)
                        )
        
        return None
    
    def _attack_enemy(self, my_planets: List, enemies: List) -> Optional[Action]:
        """Атаковать самую слабую планету врага."""
        # Сортируем врагов: предпочитаем с высоким growth и низкой защитой
        best_target = None
        best_score = -1
        best_source = None
        best_ships = 0
        
        for enemy in enemies:
            # Учитываем вражеский рост за время полёта
            for src in my_planets:
                distance = self._distance(src, enemy)
                travel_time = distance / 3.0  # transporter_speed = 3.0 по умолчанию
                enemy_ships_at_arrival = enemy.n_ships + enemy.growth_rate * travel_time
                ships_needed = int(enemy_ships_at_arrival) + 1
                
                if src.n_ships > ships_needed + 1:
                    # Атакуем если у нас минимум 1.5x преимущество
                    if src.n_ships > ships_needed * 1.5:
                        score = enemy.growth_rate / (ships_needed * max(1, distance))
                        if score > best_score:
                            best_score = score
                            best_target = enemy
                            best_source = src
                            best_ships = ships_needed + max(1, int(ships_needed * 0.3))
        
        if best_target and best_source:
            return Action(
                playerId=self.player,
                sourcePlanetId=best_source.id,
                destinationPlanetId=best_target.id,
                numShips=min(best_ships, best_source.n_ships - 1)
            )
        
        return None
    
    def _reinforce(self, my_planets: List) -> Optional[Action]:
        """Укрепить фронтовую планету."""
        if len(my_planets) < 2:
            return None
        
        # Находим планету с максимальным количеством кораблей (тыловая)
        rear = max(my_planets, key=lambda p: p.n_ships)
        if rear.n_ships < 10:
            return None
        
        # Находим фронтовую планету (ближайшую к врагу или с минимумом кораблей)
        front = min(my_planets, key=lambda p: p.n_ships)
        if front.id == rear.id:
            return None
        
        ships_to_send = min(int(rear.n_ships * 0.3), rear.n_ships - 5)
        if ships_to_send > 0:
            return Action(
                playerId=self.player,
                sourcePlanetId=rear.id,
                destinationPlanetId=front.id,
                numShips=ships_to_send
            )
        
        return None
    
    def _distance(self, p1, p2) -> float:
        return math.sqrt((p1.position.x - p2.position.x)**2 + (p1.position.y - p2.position.y)**2)
    
    def get_agent_type(self) -> str:
        return "Rule-Based Agent v1.0"