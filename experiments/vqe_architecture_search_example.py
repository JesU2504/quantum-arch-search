#!/usr/bin/env python3
"""
VQE Architecture Search Example - Using VQEArchitectEnv

This script demonstrates the full VQE pipeline using the VQEArchitectEnv:
1. An RL agent proposes quantum circuit architectures
2. Each architecture is optimized via classical parameter optimization
3. The agent is rewarded based on the optimized energy
4. Results are logged for reproducibility

This example runs on H2 (2 qubits) for quick demonstration and H4 (4 qubits)
for the realistic benchmark from ExpPlan.md Part 4.

Usage:
    python experiments/vqe_architecture_search_example.py [--molecule H2|H4]
                                                          [--episodes N]
                                                          [--output-dir DIR]

Example:
    # Quick demo on H2
    python experiments/vqe_architecture_search_example.py --molecule H2 --episodes 10

    # Full H4 benchmark
    python experiments/vqe_architecture_search_example.py --molecule H4 --episodes 50
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qas_gym.envs import VQEArchitectEnv


# Chemical accuracy threshold (1.6 mHa)
CHEMICAL_ACCURACY = 0.0016


def run_random_agent(env, n_episodes=10, seed=42):
    """
    Run random agent on VQEArchitectEnv.

    This is a baseline agent that randomly selects gates. It demonstrates
    the environment interface without requiring a trained RL agent.

    Args:
        env: VQEArchitectEnv instance
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility

    Returns:
        List of episode results
    """
    results = []
    np.random.seed(seed)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Random action (excluding DONE action for first few steps)
            if steps < 3:
                # Force at least 3 gates before allowing DONE
                action = np.random.randint(0, env.action_space.n - 1)
            else:
                action = np.random.randint(0, env.action_space.n)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        # Collect results
        result = {
            'episode': ep + 1,
            'n_gates': info.get('n_gates', 0),
            'n_cnots': info.get('n_cnots', 0),
            'initial_energy': info.get('initial_energy'),
            'optimized_energy': info.get('optimized_energy'),
            'energy_error_ha': info.get('energy_error'),
            'energy_error_mha': info.get('energy_error', 0) * 1000,
            'chemical_accuracy': info.get('chemical_accuracy_achieved', False),
            'reward': episode_reward,
        }
        results.append(result)

        print(f"Episode {ep+1:3d}: Gates={result['n_gates']:2d}, "
              f"CNOTs={result['n_cnots']:2d}, "
              f"Energy={result['optimized_energy']:.4f} Ha, "
              f"Error={result['energy_error_mha']:.2f} mHa, "
              f"Accuracy={'✓' if result['chemical_accuracy'] else '✗'}")

    return results


def run_greedy_agent(env, n_episodes=10, seed=42):
    """
    Run a simple greedy agent on VQEArchitectEnv.

    This agent tries each possible action and picks the one that results
    in the lowest energy estimate. It's a simple baseline that outperforms
    random but is much slower.

    Args:
        env: VQEArchitectEnv instance
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility

    Returns:
        List of episode results
    """
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0

        while not done:
            # Try each non-DONE action and evaluate energy
            best_action = None
            best_energy = float('inf')

            # Only evaluate up to first 3 gates, then use heuristic
            if len(env.circuit_gates) < 5:
                for action in range(env.action_space.n - 1):  # Exclude DONE
                    # Simulate adding this gate
                    test_gate = env._decode_action(action)
                    temp_gates = env.circuit_gates + [test_gate]

                    # Quick energy estimate (without full optimization)
                    if test_gate['type'] in ['Rx', 'Ry', 'Rz']:
                        # Use zero angle for quick estimate
                        temp_params = env.rotation_params.copy()
                        temp_params[len(env.circuit_gates)] = 0.0
                    else:
                        temp_params = env.rotation_params.copy()

                    # Build and evaluate circuit
                    orig_gates = env.circuit_gates
                    env.circuit_gates = temp_gates
                    circuit = env._build_circuit(temp_params)
                    energy = env.compute_energy(circuit)
                    env.circuit_gates = orig_gates

                    if energy < best_energy:
                        best_energy = energy
                        best_action = action

                action = best_action if best_action is not None else 0
            else:
                # After 5 gates, just stop
                action = env.action_space.n - 1  # DONE

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        result = {
            'episode': ep + 1,
            'n_gates': info.get('n_gates', 0),
            'n_cnots': info.get('n_cnots', 0),
            'initial_energy': info.get('initial_energy'),
            'optimized_energy': info.get('optimized_energy'),
            'energy_error_ha': info.get('energy_error'),
            'energy_error_mha': info.get('energy_error', 0) * 1000,
            'chemical_accuracy': info.get('chemical_accuracy_achieved', False),
            'reward': episode_reward,
        }
        results.append(result)

        print(f"Episode {ep+1:3d}: Gates={result['n_gates']:2d}, "
              f"CNOTs={result['n_cnots']:2d}, "
              f"Energy={result['optimized_energy']:.4f} Ha, "
              f"Error={result['energy_error_mha']:.2f} mHa, "
              f"Accuracy={'✓' if result['chemical_accuracy'] else '✗'}")

    return results


def print_summary(results, molecule, fci_energy, hf_energy):
    """Print summary statistics for experiment results."""
    print("\n" + "=" * 70)
    print(f"SUMMARY: {molecule} VQE Architecture Search")
    print("=" * 70)

    energies = [r['optimized_energy'] for r in results if r['optimized_energy'] is not None]
    errors = [r['energy_error_mha'] for r in results if r['energy_error_mha'] is not None]
    cnots = [r['n_cnots'] for r in results]
    gates = [r['n_gates'] for r in results]
    accuracy_count = sum(1 for r in results if r['chemical_accuracy'])

    print(f"Reference energies:")
    print(f"  Hartree-Fock: {hf_energy:.4f} Ha")
    print(f"  FCI (exact):  {fci_energy:.4f} Ha")
    print(f"  Correlation:  {(hf_energy - fci_energy)*1000:.2f} mHa")
    print()
    print(f"Results ({len(results)} episodes):")
    print(f"  Best energy:      {min(energies):.4f} Ha")
    print(f"  Mean energy:      {np.mean(energies):.4f} ± {np.std(energies):.4f} Ha")
    print(f"  Best error:       {min(errors):.2f} mHa")
    print(f"  Mean error:       {np.mean(errors):.2f} ± {np.std(errors):.2f} mHa")
    print(f"  Mean gate count:  {np.mean(gates):.1f} ± {np.std(gates):.1f}")
    print(f"  Mean CNOT count:  {np.mean(cnots):.1f} ± {np.std(cnots):.1f}")
    print(f"  Chemical accuracy achieved: {accuracy_count}/{len(results)} "
          f"({100*accuracy_count/len(results):.1f}%)")
    print()

    # Find best circuit
    best_idx = np.argmin(errors)
    best = results[best_idx]
    print(f"Best circuit (episode {best['episode']}):")
    print(f"  Energy: {best['optimized_energy']:.4f} Ha")
    print(f"  Error:  {best['energy_error_mha']:.2f} mHa")
    print(f"  Gates:  {best['n_gates']} (CNOTs: {best['n_cnots']})")
    print(f"  Chemical accuracy: {'✓ YES' if best['chemical_accuracy'] else '✗ NO'}")
    print("=" * 70)


def save_results(results, env, output_dir, agent_type):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'molecule': env.molecule,
            'bond_distance': env.bond_distance,
            'n_qubits': env.n_qubits,
            'max_timesteps': env.max_timesteps,
            'agent_type': agent_type,
            'reference_energies': {
                'hartree_fock': env.reference_energy,
                'fci': env.fci_energy
            }
        },
        'results': results
    }

    output_file = os.path.join(output_dir, f'vqe_{env.molecule.lower()}_{agent_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='VQE Architecture Search Example using VQEArchitectEnv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--molecule', type=str, default='H2', choices=['H2', 'H4'],
        help='Molecule to simulate (default: H2)'
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help='Number of episodes to run (default: 10)'
    )
    parser.add_argument(
        '--max-gates', type=int, default=None,
        help='Maximum gates per episode (default: 10 for H2, 15 for H4)'
    )
    parser.add_argument(
        '--agent', type=str, default='random', choices=['random', 'greedy'],
        help='Agent type: random or greedy (default: random)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save results (default: results/vqe_example_<timestamp>)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set defaults based on molecule
    if args.max_gates is None:
        args.max_gates = 10 if args.molecule == 'H2' else 15

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        args.output_dir = f'results/vqe_example_{timestamp}'

    # Print header
    print("=" * 70)
    print("VQE Architecture Search using VQEArchitectEnv")
    print("=" * 70)
    print(f"Molecule:     {args.molecule}")
    print(f"Episodes:     {args.episodes}")
    print(f"Max gates:    {args.max_gates}")
    print(f"Agent:        {args.agent}")
    print(f"Seed:         {args.seed}")
    print(f"Output dir:   {args.output_dir}")
    print()

    # Create environment with logging
    log_dir = os.path.join(args.output_dir, 'episode_logs')
    env = VQEArchitectEnv(
        molecule=args.molecule,
        max_timesteps=args.max_gates,
        log_dir=log_dir
    )

    print(f"Environment created:")
    print(f"  Qubits: {env.n_qubits}")
    print(f"  Actions: {env.action_space.n} "
          f"({env.n_rotation_actions} rotations + {env.n_cnot_actions} CNOTs + 1 DONE)")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    print(f"  HF energy: {env.reference_energy:.4f} Ha")
    print(f"  FCI energy: {env.fci_energy:.4f} Ha")
    print()

    # Run agent
    print("Running episodes...")
    print("-" * 70)
    if args.agent == 'random':
        results = run_random_agent(env, n_episodes=args.episodes, seed=args.seed)
    else:
        results = run_greedy_agent(env, n_episodes=args.episodes, seed=args.seed)

    # Print summary
    print_summary(results, args.molecule, env.fci_energy, env.reference_energy)

    # Save results
    save_results(results, env, args.output_dir, args.agent)

    # Show best circuit
    best_circuit, best_energy = env.get_best_circuit()
    if best_circuit is not None:
        print("\nBest circuit found:")
        print(best_circuit)


if __name__ == '__main__':
    main()
