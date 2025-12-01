#!/usr/bin/env python3
"""
Demo EA (Evolutionary Algorithm) agent runner for local testing.

This lightweight script simulates an EA/coevolutionary agent for classification
tasks, outputting synthetic logs in the expected JSONL format. It is intended
for testing the comparison pipeline locally without requiring a full EA
implementation.

Usage:
    python tools/run_ea_agent.py --config <config.yaml> --seed <seed> --output <output.log>

Arguments:
    --config        Path to the YAML configuration file
    --seed          Random seed for reproducibility
    --output        Output log file path (JSONL format)
    --generations   Number of generations to simulate (default: 60)
    --dry-run       Print actions without writing files

Example:
    python tools/run_ea_agent.py \\
        --config comparison/experiments/configs/ea_classification.yaml \\
        --seed 42 \\
        --output comparison/logs/ea/ea_classif_seed42.log
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone


def simulate_ea_training(seed: int, num_generations: int = 60):
    """
    Simulate EA/coevolutionary training and yield log entries.
    
    This generates synthetic data that follows the log schema expected
    by the comparison analysis tools.
    """
    random.seed(seed)
    
    # Initial accuracy starts low and improves over generations
    val_accuracy = 0.5 + random.uniform(0, 0.1)
    test_accuracy = 0.5 + random.uniform(0, 0.1)
    gate_count = 2
    circuit_depth = 2
    population_best_fitness = 0.5
    
    for generation in range(1, num_generations + 1):
        # Simulate improvement over time with noise (EA tends to improve in steps)
        if random.random() < 0.3:  # Evolutionary jumps
            improvement = random.uniform(0.01, 0.05) * (1 - val_accuracy)
            val_accuracy = min(val_accuracy + improvement, 1.0)
            test_accuracy = min(test_accuracy + improvement * random.uniform(0.9, 1.1), 1.0)
        else:
            # Small random fluctuations
            val_accuracy = min(max(val_accuracy + random.uniform(-0.01, 0.02), 0.5), 1.0)
            test_accuracy = min(max(test_accuracy + random.uniform(-0.01, 0.02), 0.5), 1.0)
        
        # Occasionally mutate circuit structure
        if random.random() < 0.15 and gate_count < 25:
            gate_count += random.randint(1, 2)
            circuit_depth = min(circuit_depth + 1, gate_count)
        
        # Population best fitness tracks best seen accuracy
        population_best_fitness = max(population_best_fitness, val_accuracy)
        
        yield {
            "eval_id": generation,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "method": "ea",
            "seed": seed,
            "best_val_accuracy": round(val_accuracy, 4),
            "best_test_accuracy": round(test_accuracy, 4),
            "gate_count": gate_count,
            "circuit_depth": circuit_depth,
            "generation": generation,
            "population_best_fitness": round(population_best_fitness, 4),
            "cum_eval_count": generation * 20,  # population_size = 20
            "wall_time_s": round(generation * 0.5, 2),
            "notes": "Demo EA run"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Demo EA agent runner for local testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", required=True, help="Output log file path")
    parser.add_argument("--generations", type=int, default=60, help="Number of generations")
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
    print(f"Generations: {args.generations}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would generate synthetic EA training logs")
        return 0
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate and write log entries
    print(f"Starting demo EA training simulation...")
    entries_written = 0
    
    with open(args.output, "w") as f:
        for entry in simulate_ea_training(args.seed, args.generations):
            f.write(json.dumps(entry) + "\n")
            entries_written += 1
            
            # Progress reporting
            if entries_written % 10 == 0:
                print(f"  Generation {entry['generation']}: val_acc={entry['best_val_accuracy']:.4f}, "
                      f"test_acc={entry['best_test_accuracy']:.4f}, gates={entry['gate_count']}")
    
    print()
    print(f"Demo EA training complete!")
    print(f"  Total generations: {entries_written}")
    print(f"  Log file: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
