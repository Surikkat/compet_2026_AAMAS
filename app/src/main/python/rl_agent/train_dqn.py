"""
DQN Training script for Planet Wars Pairwise RL Agent (V2 — per-planet LLM).

Trains the DQN network by running self-play episodes against a
GreedyHeuristicAgent opponent. During training, the V2 LLM is called
every LLM_CALL_INTERVAL ticks to provide per-planet strategy letters
(A/P/E/N), so the RL agent learns to follow granular orders.

Usage:
    python rl_agent/train_dqn.py [--episodes N] [--save-path PATH]
"""

import os
import sys
import argparse
import random
import time
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from core.game_state import GameState, GameParams, Player, Action
from core.game_state_factory import GameStateFactory
from core.forward_model import ForwardModel
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.random_agents import CarefulRandomAgent

from rl_agent.state_encoder import StateEncoder, FEATURE_DIM, DEFAULT_LLM_VEC, LETTER_TO_VEC
from rl_agent.action_decoder import ActionDecoder, NUM_ACTIONS
from rl_agent.dqn_network import DQNNetwork, DEVICE
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.reward import RewardCalculator

import json

# ── Mock LLM (Dataset Cache) ─────────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), "llm_cache.json")


# ──────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
BUFFER_CAPACITY = 50_000
TARGET_UPDATE_FREQ = 500       # steps between target-net syncs
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000     # linear ε decay over this many steps
TRAIN_START = 1000             # fill buffer before training begins
LOG_INTERVAL = 50              # episodes between log prints
SAVE_INTERVAL = 200            # episodes between checkpoint saves
MAX_TICKS_TRAIN = 500          # shorter episodes for faster training
LLM_CALL_INTERVAL = 100       # ticks between LLM API calls (~5 game-seconds)


def random_game_params() -> GameParams:
    """Sample game params matching AAMAS 2026 ranges."""
    return GameParams(
        num_planets=random.randint(10, 30),
        initial_neutral_ratio=random.uniform(0.25, 0.35),
        min_growth_rate=0.02,
        max_growth_rate=random.uniform(0.1, 0.2),
        transporter_speed=random.uniform(2.0, 5.0),
        max_ticks=MAX_TICKS_TRAIN,
    )


def select_action(
    q_values: torch.Tensor,
    epsilon: float,
    num_pairs: int,
) -> tuple:
    """
    Epsilon-greedy action selection over all (pair, fraction) combinations.

    Args:
        q_values: shape (num_pairs, NUM_ACTIONS)
        epsilon: exploration probability
        num_pairs: number of valid pairs

    Returns:
        (pair_idx, action_idx)
    """
    if random.random() < epsilon:
        pair_idx = random.randint(0, num_pairs - 1)
        action_idx = random.randint(0, NUM_ACTIONS - 1)
        return pair_idx, action_idx

    # Greedy: find global max Q
    flat_idx = q_values.argmax().item()
    pair_idx = flat_idx // NUM_ACTIONS
    action_idx = flat_idx % NUM_ACTIONS
    return pair_idx, action_idx


def train_step(
    policy_net: DQNNetwork,
    target_net: DQNNetwork,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
) -> float:
    """
    One gradient step of DQN training.

    Returns:
        Loss value (float).
    """
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    # Q(s, a) for the actions that were actually taken
    q_values = policy_net(states)  # (B, NUM_ACTIONS)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # Target: r + γ * max_a' Q_target(s', a')
    with torch.no_grad():
        next_q = target_net(next_states)  # (B, NUM_ACTIONS)
        max_next_q = next_q.max(dim=1).values  # (B,)
        # Clamp target Q-values to prevent runaway
        max_next_q = max_next_q.clamp(-100, 100)
        target = rewards.clamp(-10, 10) + GAMMA * max_next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(q_sa, target)

    # Skip if loss is NaN (defensive)
    if torch.isnan(loss):
        return float('nan')

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def extract_planet_state_vector(planet, state: GameState, player: Player) -> np.ndarray:
    """Extract 3-dimensional continuous state vector for a single planet."""
    enemy_planets = [p for p in state.planets if p.owner == player.opponent()]
    if not enemy_planets:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    avg_enemy_x = sum(p.position.x for p in enemy_planets) / len(enemy_planets)
    avg_enemy_y = sum(p.position.y for p in enemy_planets) / len(enemy_planets)
    
    my_planets = [p for p in state.planets if p.owner == player]
    if not my_planets:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    distances = [((p.position.x - avg_enemy_x)**2 + (p.position.y - avg_enemy_y)**2)**0.5 for p in my_planets]
    threshold = sum(distances) / len(distances)
    
    dist = ((planet.position.x - avg_enemy_x)**2 + (planet.position.y - avg_enemy_y)**2)**0.5
    is_frontline = 1.0 if dist <= threshold else 0.0
    
    my_ships = sum(p.n_ships for p in state.planets if p.owner == player) + 1e-5
    ship_ratio = planet.n_ships / my_ships
    
    max_growth = max(p.growth_rate for p in state.planets) + 1e-5
    growth_ratio = planet.growth_rate / max_growth
    
    return np.array([is_frontline, ship_ratio, growth_ratio], dtype=np.float32)

def _load_llm_cache() -> list:
    """Load pre-generated dataset of LLM strategies."""
    if not os.path.exists(CACHE_PATH):
        print(f"[train] WARNING: Cache file {CACHE_PATH} not found.")
        print(f"[train] Please run generate_llm_cache.py first. Using random commands.")
        return []
    with open(CACHE_PATH, "r") as f:
        data = json.load(f)
    print(f"[train] Loaded {len(data)} cached LLM strategies.")
    
    # Convert lists to numpy arrays for fast distance computation
    for item in data:
        item["state_vector"] = np.array(item["state_vector"], dtype=np.float32)
    return data

def _get_cached_strategy(cache: list, state_vec: np.ndarray) -> str:
    """Find nearest strategy in cache using Euclidean distance."""
    if not cache:
        # Fallback to default action
        return "N"
    
    min_dist = float('inf')
    best_strategy = cache[0]["strategy"]
    for item in cache:
        dist = np.linalg.norm(item["state_vector"] - state_vec)
        if dist < min_dist:
            min_dist = dist
            best_strategy = item["strategy"]
    return best_strategy


def run_training(
    num_episodes: int,
    save_path: str = os.path.join(os.path.dirname(__file__), "checkpoints", "dqn_latest.pt")
) -> None:
    """Main training loop."""

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"[train] Using device: {DEVICE}")

    # Networks — move to GPU/MPS
    policy_net = DQNNetwork().to(DEVICE)
    target_net = DQNNetwork().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    decoder = ActionDecoder()

    # Load Mock LLM cache
    llm_cache = _load_llm_cache()

    global_step = 0
    episode_rewards: List[float] = []
    episode_wins: List[bool] = []
    recent_losses: List[float] = []
    llm_calls_total = 0

    t_start = time.time()

    pbar = tqdm(range(1, num_episodes + 1), desc="Training DQN (V2)")
    for episode in pbar:
        # --- Setup episode ------------------------------------------------
        params = random_game_params()
        encoder = StateEncoder(params)
        reward_calc = RewardCalculator(Player.Player1)

        state = GameStateFactory(params).create_game()
        fm = ForwardModel(state.model_copy(deep=True), params)

        # Opponent (alternate between greedy and random for diversity)
        if random.random() < 0.7:
            opponent = GreedyHeuristicAgent()
        else:
            opponent = CarefulRandomAgent()
        opponent.prepare_to_play_as(Player.Player2, params)

        reward_calc.reset(fm.state)

        # Per-planet LLM strategy — letters are stored directly
        # Format: {planet_id: 'A'/'P'/'E'/'N'}
        llm_strategy_letters: Dict[int, str] = {}
        # Converted to vectors for the encoder
        # Format: {planet_id: [1.0, 0.0, 0.0, 0.0]}
        llm_strategy: Dict[int, List[float]] = {}

        ep_reward = 0.0
        epsilon = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS)
        tick_count = 0

        # --- Episode loop -------------------------------------------------
        while not fm.is_terminal():
            # ── Call V2 Mock LLM periodically (per-planet) ─────────────────
            if tick_count % LLM_CALL_INTERVAL == 0:
                for p in fm.state.planets:
                    if p.owner == Player.Player1:
                        p_vec = extract_planet_state_vector(p, fm.state, Player.Player1)
                        letter = _get_cached_strategy(llm_cache, p_vec)
                        llm_strategy_letters[p.id] = letter
                        llm_strategy[p.id] = LETTER_TO_VEC.get(letter, DEFAULT_LLM_VEC)
                llm_calls_total += 1

            # Encode all valid pairs
            features, pair_info = encoder.encode_all_pairs(
                fm.state, Player.Player1, llm_strategy
            )

            if len(pair_info) == 0:
                # No valid source planets — do nothing
                our_action = Action.do_nothing()
                chosen_features = None
                action_idx = 0
                source = None
            else:
                # Forward pass (on GPU)
                with torch.no_grad():
                    q_values = policy_net(
                        torch.FloatTensor(features).to(DEVICE)
                    )

                pair_idx, action_idx = select_action(
                    q_values, epsilon, len(pair_info)
                )
                source, target = pair_info[pair_idx]
                our_action = decoder.decode(
                    action_idx, source, target, Player.Player1
                )
                chosen_features = features[pair_idx]

            # Opponent acts
            opp_state = fm.state.model_copy(deep=True)
            opp_action = opponent.get_action(opp_state)

            # Step the simulation
            fm.step({
                Player.Player1: our_action,
                Player.Player2: opp_action,
            })

            done = fm.is_terminal()
            winner = fm.get_leader() if done else None

            # Determine LLM command for this source planet (per-planet letter)
            llm_cmd = ""
            if source is not None:
                llm_cmd = llm_strategy_letters.get(source.id, "")

            reward = reward_calc.compute(
                fm.state, done, winner, our_action, llm_cmd
            )
            ep_reward += reward

            # Store transition (only when we made a real choice)
            if chosen_features is not None:
                # Look up planets by ID in the NEW state (old refs may be stale)
                new_source = fm.state.planets[source.id]
                new_target = fm.state.planets[target.id]
                next_features = encoder.encode_pair(
                    new_source, new_target, fm.state, Player.Player1, llm_strategy
                )
                # Guard against NaN
                if not (np.isnan(chosen_features).any() or np.isnan(next_features).any()):
                    buffer.push(chosen_features, action_idx, reward, next_features, done)

            global_step += 1
            tick_count += 1

            # Train on mini-batch
            if len(buffer) >= TRAIN_START and len(buffer) >= BATCH_SIZE:
                loss = train_step(policy_net, target_net, buffer, optimizer)
                if not math.isnan(loss):
                    recent_losses.append(loss)

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # --- End of episode -----------------------------------------------
        final_leader = fm.get_leader()
        won = final_leader == Player.Player1
        episode_rewards.append(ep_reward)
        episode_wins.append(won)

        # --- Logging ------------------------------------------------------
        if episode % LOG_INTERVAL == 0:
            recent_wr = (
                sum(episode_wins[-LOG_INTERVAL:]) / LOG_INTERVAL * 100
            )
            avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
            avg_loss = np.mean(recent_losses[-200:]) if recent_losses else 0
            eps = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS)

            pbar.set_postfix({
                'WR': f'{recent_wr:.1f}%',
                'Reward': f'{avg_reward:.1f}',
                'Loss': f'{avg_loss:.3f}',
                'eps': f'{eps:.2f}',
                'LLM': llm_calls_total,
            })

        # --- Checkpointing ------------------------------------------------
        if episode % SAVE_INTERVAL == 0:
            torch.save(policy_net.state_dict(), save_path)

    pbar.close()

    # Final save
    torch.save(policy_net.state_dict(), save_path)
    print(f"\nTraining complete. Final checkpoint: {save_path}")
    print(f"Total steps: {global_step}")
    print(f"Total LLM API calls: {llm_calls_total}")
    total_wr = sum(episode_wins) / len(episode_wins) * 100
    print(f"Overall win rate: {total_wr:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Planet Wars (V2 per-planet)")
    parser.add_argument(
        "--episodes", type=int, default=3000,
        help="Number of training episodes (default: 3000)",
    )
    parser.add_argument(
        "--save-path", type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "dqn_latest.pt"),
        help="Path for saving model checkpoints",
    )
    args = parser.parse_args()

    run_training(args.episodes, args.save_path)
