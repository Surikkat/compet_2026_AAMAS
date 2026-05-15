import argparse
import time

from core.game_state import GameParams, Player
from core.unified_game_runner import UnifiedGameRunner
from agents.fully_observable_agent_adapter import as_unified
from agents.random_agents import CarefulRandomAgent, PureRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from rule_based_agent import RuleBasedAgent

# Все агенты для сравнения
ALL_AGENTS = {
    "PureRandom": lambda: PureRandomAgent(),
    "CarefulRandom": lambda: CarefulRandomAgent(),
    "GreedyHeuristic": lambda: GreedyHeuristicAgent(),
    "RuleBased": lambda: RuleBasedAgent(),
}

class DoNothingAgent:
    """Агент, который ничего не делает."""
    def prepare_to_play_as(self, player, params, opponent=None):
        self.player = player
        return "DoNothing"
    def get_action(self, state):
        return __import__('core.game_state', fromlist=['Action']).Action.do_nothing()
    def get_agent_type(self):
        return "DoNothing"

def run_tournament(n_games: int = 20, checkpoint_path: str = None):
    params = GameParams(num_planets=20)
    
    agents = list(ALL_AGENTS.keys())
    
    print(f"{'='*80}")
    print(f"PLANET WARS TOURNAMENT ({n_games} games per match)")
    print(f"{'='*80}\n")
    
    results = {}
    
    for i, name1 in enumerate(agents):
        for name2 in agents:
            if name1 >= name2:  # Избегаем дубликатов и self-play
                continue
            
            agent1 = ALL_AGENTS[name1]()
            agent2 = ALL_AGENTS[name2]()
            
            # Адаптируем агентов для UnifiedGameRunner
            if hasattr(agent1, 'get_action'):
                a1 = as_unified(agent1) if not isinstance(agent1, DoNothingAgent) else agent1
            if hasattr(agent2, 'get_action'):
                a2 = as_unified(agent2) if not isinstance(agent2, DoNothingAgent) else agent2
            
            print(f"{name1:20} vs {name2:20} ... ", end="", flush=True)
            
            try:
                runner = UnifiedGameRunner(a1, a2, params, partial_observability=False)
                t0 = time.time()
                scores = runner.run_games(n_games)
                elapsed = time.time() - t0
                
                wins1 = scores.get(Player.Player1, 0)
                wins2 = scores.get(Player.Player2, 0)
                draws = scores.get(Player.Neutral, 0)
                
                results[(name1, name2)] = (wins1, wins2, draws)
                winrate = wins1 / n_games * 100
                print(f"{wins1}-{wins2} ({winrate:.0f}%) [{elapsed:.1f}s]")
            except Exception as e:
                print(f"ERROR: {e}")
                results[(name1, name2)] = (0, 0, 0)
    
    # Итоговая таблица
    print(f"\n{'='*80}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*80}")
    print(f"{'Agent':20} {'Wins':>6} {'Losses':>6} {'Winrate':>8}")
    print("-"*42)
    
    total_stats = {name: {'wins': 0, 'losses': 0} for name in agents}
    
    for (name1, name2), (w1, w2, d) in results.items():
        total_stats[name1]['wins'] += w1
        total_stats[name1]['losses'] += w2
        total_stats[name2]['wins'] += w2
        total_stats[name2]['losses'] += w1
    
    sorted_agents = sorted(total_stats.items(), 
                          key=lambda x: x[1]['wins'] / max(1, x[1]['wins'] + x[1]['losses']),
                          reverse=True)
    
    for name, stats in sorted_agents:
        total = stats['wins'] + stats['losses']
        winrate = stats['wins'] / total * 100 if total > 0 else 0
        print(f"{name:20} {stats['wins']:6} {stats['losses']:6} {winrate:7.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_tournament(args.games, args.checkpoint)
