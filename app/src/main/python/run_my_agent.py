import argparse
import time

from core.game_state import GameParams, Player
from core.unified_game_runner import UnifiedGameRunner
from agents.fully_observable_agent_adapter import as_unified
from agents.random_agents import CarefulRandomAgent
from my_agent import MyPythonAgent

def run_evaluation(n_games: int, checkpoint_path: str = None):
    params = GameParams(num_planets=20)
    agent1 = MyPythonAgent(checkpoint_path=checkpoint_path)
    agent2 = CarefulRandomAgent()

    runner = UnifiedGameRunner(
        as_unified(agent1), as_unified(agent2), params, partial_observability=False
    )

    print(f"Running {n_games} games: LLM vs CarefulRandom...")
    t0 = time.time()
    scores = runner.run_games(n_games)
    elapsed = time.time() - t0

    wins = scores.get(Player.Player1, 0)
    losses = scores.get(Player.Player2, 0)
    draws = scores.get(Player.Neutral, 0)

    print(f"\nResults ({n_games} games, {elapsed:.1f}s):")
    print(f"  LLM wins:      {wins} ({wins/n_games*100:.1f}%)")
    print(f"  Careful wins:  {losses} ({losses/n_games*100:.1f}%)")
    print(f"  Draws:         {draws}")
    print(f"  Time per game: {elapsed/n_games*1000:.0f} ms")

    if hasattr(agent1, 'llm_agent') and hasattr(agent1.llm_agent, 'get_stats'):
        stats = agent1.llm_agent.get_stats()
        print(f"\nTiming Stats:")
        print(f"  LLM API calls:  {stats['llm_calls']}")
        print(f"  Cache hits:     {stats['cache_hits']}")
        print(f"  Fallbacks:      {stats['fallback_calls']}")
        print(f"  Avg action:     {stats['avg_action_time_ms']:.3f} ms")
        print(f"  Max action:     {stats['max_action_time_ms']:.3f} ms")
        print(f"  Under 50ms:     {stats['under_50ms_pct']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(args.games, args.checkpoint)
