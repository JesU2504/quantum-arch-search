#!/usr/bin/env python3
"""
Demo EA agent runner for exercising the comparison pipeline end-to-end.

This is a minimal demonstration runner that generates synthetic log data
matching the expected schema. It is NOT a full EA/coevolution implementation,
but allows testing the comparison infrastructure without running actual evolution.

Usage:
    python tools/run_ea_agent.py --config CONFIG --seed SEED --output OUTPUT

Arguments:
    --config: Path to YAML config file (used for metadata)
    --seed: Random seed for reproducibility
    --output: Path to output log file (JSONL format)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo EA agent runner for comparison pipeline testing"
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
        "--generations",
        type=int,
        default=10,
        help="Number of generations to simulate (default: 10)",
    )
    return parser.parse_args()


def generate_demo_metrics(seed: int, generation: int, total_generations: int) -> dict:
    """
    Generate synthetic metrics that simulate EA training progress.
    
    Metrics follow a realistic evolutionary improvement curve with noise.
    EA typically shows more variance than DRL due to population-based search.
    
    Args:
        seed: Random seed for reproducibility
        generation: Current generation number
        total_generations: Total number of generations
    
    Returns:
        Dictionary with metrics matching the log schema
    """
    # Set seed for reproducibility
    random.seed(seed + generation * 7)  # Different multiplier than DRL for variety
    
    # Simulate accuracy improvement - EA typically has more variance
    progress = generation / total_generations
    # EA often shows step-wise improvement as better individuals are found
    base_accuracy = 0.55 + 0.35 * (1 - (1 - progress) ** 1.5)  # Starts at 0.55
    noise = random.gauss(0, 0.03)  # More noise than DRL
    test_accuracy = max(0.0, min(1.0, base_accuracy + noise))
    train_accuracy = test_accuracy + random.gauss(0.02, 0.01)
    train_accuracy = max(0.0, min(1.0, train_accuracy))
    
    # EA tends to find more compact solutions
    base_gates = 4 + int(progress * 8)
    gate_noise = random.randint(-2, 3)
    gate_count = max(1, base_gates + gate_noise)
    
    # Circuit depth
    circuit_depth = max(1, gate_count - random.randint(0, 3))
    
    # Population best fitness (similar to accuracy but population-based)
    population_best_fitness = test_accuracy + random.gauss(0, 0.01)
    population_best_fitness = max(0.0, min(1.0, population_best_fitness))
    
    return {
        "eval_id": generation,
        "timestamp": datetime.now().isoformat(),
        "method": "ea",
        "seed": seed,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "best_val_accuracy": round(test_accuracy, 4),
        "best_test_accuracy": round(test_accuracy, 4),
        "gate_count": gate_count,
        "circuit_depth": circuit_depth,
        "generation": generation,
        "population_best_fitness": round(population_best_fitness, 4),
        "cum_eval_count": (generation + 1) * 20,  # ~20 evals per generation
        "wall_time_s": round((generation + 1) * 0.8, 2),  # Simulate 0.8s per gen
        "notes": "demo_run",
    }


def main() -> int:
    """Run the demo EA agent."""
    args = parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file not found: {args.config}", file=sys.stderr)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Demo EA Agent ===")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Generations: {args.generations}")
    print()
    
    # Generate and write log entries
    logs = []
    for gen in range(args.generations):
        metrics = generate_demo_metrics(args.seed, gen, args.generations)
        logs.append(metrics)
        print(f"Generation {gen + 1}/{args.generations}: "
              f"test_acc={metrics['test_accuracy']:.3f}, "
              f"gates={metrics['gate_count']}, "
              f"pop_best={metrics['population_best_fitness']:.3f}")
        
        # Small delay to simulate evolution time
        time.sleep(0.01)
    
    # Write JSONL output
    with open(output_path, 'w') as f:
        for entry in logs:
            f.write(json.dumps(entry) + '\n')
    
    print()
    print(f"Results written to: {args.output}")
    
    # Also write to stdout for the tee'd log file (comparison/logs/ea/...)
    final_metrics = logs[-1]
    print()
    print(f"=== Final Results ===")
    print(f"Best test accuracy: {max(m['test_accuracy'] for m in logs):.4f}")
    print(f"Final gate count: {final_metrics['gate_count']}")
    print(f"Total generations: {args.generations}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
