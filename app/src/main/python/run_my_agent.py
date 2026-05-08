import argparse
import time

from core.game_state import GameParams, Player
from core.unified_game_runner import UnifiedGameRunner
from agents.fully_observable_agent_adapter import as_unified
from agents.random_agents import CarefulRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent


from my_agent import MyPythonAgent


def run_evaluation(n_games: int, checkpoint_path: str = None):
    params = GameParams(num_planets=20)

    agent1 = MyPythonAgent(checkpoint_path=checkpoint_path)
    agent2 = CarefulRandomAgent()

    runner = UnifiedGameRunner(
        as_unified(agent1),
        as_unified(agent2),
        params,
        partial_observability=False,
    )

    print(f"Running {n_games} games: DQN vs GreedyHeuristic...")
    t0 = time.time()
    scores = runner.run_games(n_games)
    elapsed = time.time() - t0

    wins = scores.get(Player.Player1, 0)
    losses = scores.get(Player.Player2, 0)
    draws = scores.get(Player.Neutral, 0)

    print(f"\nResults ({n_games} games, {elapsed:.1f}s):")
    print(f"  DQN wins:      {wins} ({wins/n_games*100:.1f}%)")
    print(f"  Greedy wins:   {losses} ({losses/n_games*100:.1f}%)")
    print(f"  Draws:         {draws}")
    print(f"  Time per game: {elapsed/n_games*1000:.0f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DQN agent locally")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    run_evaluation(args.games, args.checkpoint)