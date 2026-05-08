"""
DQN Training script for Planet Wars Pairwise RL Agent.

Trains the DQN network by running self-play episodes against a
GreedyHeuristicAgent opponent. Randomizes game parameters to match
AAMAS 2026 competition settings.

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

from core.game_state import GameState, GameParams, Player, Action
from core.game_state_factory import GameStateFactory
from core.forward_model import ForwardModel
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.random_agents import CarefulRandomAgent

from rl_agent.state_encoder import StateEncoder, FEATURE_DIM
from rl_agent.action_decoder import ActionDecoder, NUM_ACTIONS
from rl_agent.dqn_network import DQNNetwork
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.reward import RewardCalculator


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
MAX_TICKS_TRAIN = 2000          # shorter episodes for faster training


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


def run_training(num_episodes: int, save_path: str) -> None:
    """Main training loop."""

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Networks
    policy_net = DQNNetwork()
    target_net = DQNNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    decoder = ActionDecoder()

    global_step = 0
    episode_rewards: List[float] = []
    episode_wins: List[bool] = []
    recent_losses: List[float] = []

    t_start = time.time()
    
    pbar = tqdm(range(1, num_episodes + 1), desc="Training DQN")
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

        # Mock LLM strategy (uniform)
        llm_strategy: Dict[int, List[float]] = {}

        ep_reward = 0.0
        epsilon = max(EPS_END, EPS_START - global_step / EPS_DECAY_STEPS)

        # --- Episode loop -------------------------------------------------
        while not fm.is_terminal():
            # Encode all valid pairs
            features, pair_info = encoder.encode_all_pairs(
                fm.state, Player.Player1, llm_strategy
            )

            if len(pair_info) == 0:
                # No valid source planets — do nothing
                our_action = Action.do_nothing()
                chosen_features = None
                action_idx = 0
            else:
                # Forward pass
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(features))

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
            reward = reward_calc.compute(fm.state, done, winner)
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
                'eps': f'{eps:.2f}'
            })

        # --- Checkpointing ------------------------------------------------
        if episode % SAVE_INTERVAL == 0:
            torch.save(policy_net.state_dict(), save_path)
            # Optional: log checkpoint via tqdm.write if needed:
            # tqdm.write(f"  → Checkpoint saved to {save_path}")

    pbar.close()
    
    # Final save
    torch.save(policy_net.state_dict(), save_path)
    print(f"\nTraining complete. Final checkpoint: {save_path}")
    print(f"Total steps: {global_step}")
    total_wr = sum(episode_wins) / len(episode_wins) * 100
    print(f"Overall win rate: {total_wr:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Planet Wars")
    parser.add_argument(
        "--episodes", type=int, default=3000,
        help="Number of training episodes (default: 3000)",
    )
    parser.add_argument(
        "--save-path", type=str,
        default="rl_agent/checkpoints/dqn_latest.pt",
        help="Path for saving model checkpoints",
    )
    args = parser.parse_args()

    run_training(args.episodes, args.save_path)