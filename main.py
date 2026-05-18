import asyncio
import sys
sys.path.insert(0, 'app/src/main/python')  # Добавляем папку с кодом в PYTHONPATH

from client_server.game_agent_server import GameServerAgent
from agents.my_greedy_heuristic import MyGreedyHeuristicAgent

if __name__ == "__main__":
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=MyGreedyHeuristicAgent()).start())