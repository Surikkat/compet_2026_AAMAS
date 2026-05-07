from core.game_state import GameParams
from core.unified_game_runner import UnifiedGameRunner
from agents.fully_observable_agent_adapter import as_unified
from agents.random_agents import CarefulRandomAgent
from my_agent_llm import ObserverAgent

if __name__ == "__main__":
    params = GameParams(num_planets=20)
    agent1 = as_unified(ObserverAgent())
    agent2 = as_unified(CarefulRandomAgent())
    runner = UnifiedGameRunner(agent1, agent2, params, partial_observability=False)
    result = runner.run_game()
    print(f"Победитель: {result.get_leader()}")
    print(result.status_string())