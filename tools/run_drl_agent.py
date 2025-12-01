#!/usr/bin/env python3
"""
Demo DRL (Deep Reinforcement Learning) agent runner for local testing.

This lightweight script simulates a DRL agent for classification tasks,
outputting synthetic logs in the expected JSONL format. It is intended
for testing the comparison pipeline locally without requiring a full DRL
implementation.

Usage:
    python tools/run_drl_agent.py --config <config.yaml> --seed <seed> --output <output.log>

Arguments:
    --config    Path to the YAML configuration file
    --seed      Random seed for reproducibility
    --output    Output log file path (JSONL format)
    --episodes  Number of episodes to simulate (default: 100)
    --dry-run   Print actions without writing files

Example:
    python tools/run_drl_agent.py \\
        --config comparison/experiments/configs/drl_classification.yaml \\
        --seed 42 \\
        --output comparison/logs/drl/drl_classif_seed42.log
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone


def simulate_drl_training(seed: int, num_episodes: int = 100):
    """
    Simulate DRL training and yield log entries.
    
    This generates synthetic data that follows the log schema expected
    by the comparison analysis tools.
    """
    random.seed(seed)
    
    # Initial accuracy starts low and improves over episodes
    val_accuracy = 0.5 + random.uniform(0, 0.1)
    test_accuracy = 0.5 + random.uniform(0, 0.1)
    gate_count = 1
    circuit_depth = 1
    
    for episode in range(1, num_episodes + 1):
        # Simulate improvement over time with noise
        improvement = random.uniform(0, 0.02) * (1 - val_accuracy)
        val_accuracy = min(val_accuracy + improvement, 1.0)
        test_accuracy = min(test_accuracy + improvement * random.uniform(0.9, 1.1), 1.0)
        
        # Occasionally add gates (simulating architecture search)
        if random.random() < 0.2 and gate_count < 20:
            gate_count += 1
            circuit_depth = min(circuit_depth + 1, gate_count)
        
        # Calculate episode reward based on accuracy and gate penalty
        reward = val_accuracy - 0.01 * gate_count
        
        yield {
            "eval_id": episode,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "drl",
            "seed": seed,
            "best_val_accuracy": round(val_accuracy, 4),
            "best_test_accuracy": round(test_accuracy, 4),
            "gate_count": gate_count,
            "circuit_depth": circuit_depth,
            "episode_reward": round(reward, 4),
            "cum_eval_count": episode,
            "wall_time_s": round(episode * 0.1, 2),
            "notes": "Demo DRL run"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Demo DRL agent runner for local testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", required=True, help="Output log file path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    
    args = parser.parse_args()
    
    # Verify config file exists (informational only in demo mode)
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found: {args.config}", file=sys.stderr)
        print("Continuing with demo mode (synthetic data)...", file=sys.stderr)
    else:
        print(f"Config: {args.config}")
    
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Episodes: {args.episodes}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would generate synthetic DRL training logs")
        return 0
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate and write log entries
    print(f"Starting demo DRL training simulation...")
    entries_written = 0
    
    with open(args.output, "w") as f:
        for entry in simulate_drl_training(args.seed, args.episodes):
            f.write(json.dumps(entry) + "\n")
            entries_written += 1
            
            # Progress reporting
            if entries_written % 10 == 0:
                print(f"  Episode {entry['eval_id']}: val_acc={entry['best_val_accuracy']:.4f}, "
                      f"test_acc={entry['best_test_accuracy']:.4f}, gates={entry['gate_count']}")
    
    print()
    print(f"Demo DRL training complete!")
    print(f"  Total episodes: {entries_written}")
    print(f"  Log file: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
