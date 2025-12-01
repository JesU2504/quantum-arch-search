#!/usr/bin/env python3
"""
Demo DRL agent runner for exercising the comparison pipeline end-to-end.

This is a minimal demonstration runner that generates synthetic log data
matching the expected schema. It is NOT a full DRL implementation, but
allows testing the comparison infrastructure without running actual training.

Usage:
    python tools/run_drl_agent.py --config CONFIG --seed SEED --output OUTPUT

Arguments:
    --config: Path to YAML config file (used for metadata)
    --seed: Random seed for reproducibility
    --output: Path to output log file (JSONL format)
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo DRL agent runner for comparison pipeline testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output log file (JSONL format)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to simulate (default: 10)",
    )
    return parser.parse_args()


def generate_demo_metrics(seed: int, episode: int, total_episodes: int) -> dict:
    """
    Generate synthetic metrics that simulate DRL agent training progress.
    
    Metrics follow a realistic improvement curve with noise.
    
    Args:
        seed: Random seed for reproducibility
        episode: Current episode number
        total_episodes: Total number of episodes
    
    Returns:
        Dictionary with metrics matching the log schema
    """
    # Set seed for reproducibility
    random.seed(seed + episode)
    
    # Simulate accuracy improvement with diminishing returns
    progress = episode / total_episodes
    base_accuracy = 0.5 + 0.4 * (1 - (1 - progress) ** 2)  # Starts at 0.5, approaches 0.9
    noise = random.gauss(0, 0.02)
    test_accuracy = max(0.0, min(1.0, base_accuracy + noise))
    train_accuracy = test_accuracy + random.gauss(0.01, 0.01)  # Train usually slightly higher
    train_accuracy = max(0.0, min(1.0, train_accuracy))
    
    # Simulate gate count - starts low, grows then stabilizes
    base_gates = 5 + int(progress * 10)
    gate_noise = random.randint(-1, 2)
    gate_count = max(1, base_gates + gate_noise)
    
    # Circuit depth correlates with gate count but not always equal
    circuit_depth = max(1, gate_count - random.randint(0, 2))
    
    # Episode reward follows accuracy improvement
    episode_reward = (test_accuracy - 0.5) * 2 - 0.01 * gate_count
    
    return {
        "eval_id": episode,
        "timestamp": datetime.now().isoformat(),
        "method": "drl",
        "seed": seed,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "best_val_accuracy": round(test_accuracy, 4),
        "best_test_accuracy": round(test_accuracy, 4),
        "gate_count": gate_count,
        "circuit_depth": circuit_depth,
        "episode_reward": round(episode_reward, 4),
        "cum_eval_count": episode + 1,
        "wall_time_s": round((episode + 1) * 0.5, 2),  # Simulate 0.5s per episode
        "notes": "demo_run",
    }


def main() -> int:
    """Run the demo DRL agent."""
    args = parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file not found: {args.config}", file=sys.stderr)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Demo DRL Agent ===")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Episodes: {args.episodes}")
    print()
    
    # Generate and write log entries
    logs = []
    for episode in range(args.episodes):
        metrics = generate_demo_metrics(args.seed, episode, args.episodes)
        logs.append(metrics)
        print(f"Episode {episode + 1}/{args.episodes}: "
              f"test_acc={metrics['test_accuracy']:.3f}, "
              f"gates={metrics['gate_count']}, "
              f"reward={metrics['episode_reward']:.3f}")
        
        # Small delay to simulate training time
        time.sleep(0.01)
    
    # Write JSONL output
    with open(output_path, 'w') as f:
        for entry in logs:
            f.write(json.dumps(entry) + '\n')
    
    print()
    print(f"Results written to: {args.output}")
    
    # Also write to stdout for the tee'd log file (comparison/logs/drl/...)
    final_metrics = logs[-1]
    print()
    print(f"=== Final Results ===")
    print(f"Best test accuracy: {max(m['test_accuracy'] for m in logs):.4f}")
    print(f"Final gate count: {final_metrics['gate_count']}")
    print(f"Total episodes: {args.episodes}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
