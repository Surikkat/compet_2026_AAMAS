import argparse
import time

from core.game_state import GameParams, Player
from core.unified_game_runner import UnifiedGameRunner
from agents.fully_observable_agent_adapter import as_unified
from agents.random_agents import CarefulRandomAgent, PureRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from my_greedy_heuristic import MyGreedyHeuristicAgent
from rule_based_agent import RuleBasedAgent

ALL_AGENTS = {
    "PureRandom": lambda: PureRandomAgent(),
    "CarefulRandom": lambda: CarefulRandomAgent(),
    "GreedyHeuristic": lambda: GreedyHeuristicAgent(),
    "MyGreedy": lambda: MyGreedyHeuristicAgent(),
    "RuleBased": lambda: RuleBasedAgent(),
}

def run_tournament(n_games: int = 20):
    params = GameParams(num_planets=20)
    agents = list(ALL_AGENTS.keys())
    
    print(f"{'='*80}")
    print(f"PLANET WARS TOURNAMENT ({n_games} games per match)")
    print(f"{'='*80}\n")
    
    results = {}
    
    for i, name1 in enumerate(agents):
        for name2 in agents:
            if name1 >= name2:
                continue
            
            agent1 = ALL_AGENTS[name1]()
            agent2 = ALL_AGENTS[name2]()
            
            print(f"{name1:20} vs {name2:20} ... ", end="", flush=True)
            
            try:
                a1 = as_unified(agent1)
                a2 = as_unified(agent2)
                
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
    args = parser.parse_args()
    run_tournament(args.games)