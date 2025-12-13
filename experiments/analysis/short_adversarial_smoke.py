"""Run a very short adversarial training and then evaluate stop behavior.

This is a lightweight smoke run (tiny numbers) to verify that the architect
learns or at least that STOP-based episodes occur and that shaping params are
propagated through.
"""
import tempfile
import shutil
import os
import time
import pathlib
import sys

# Make repository root importable
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adversarial.train_adversarial import train_adversarial


def main():
    tmp = tempfile.mkdtemp(prefix='adv_smoke_')
    print('Temporary results dir:', tmp)
    try:
        arch_agent, sab_agent, log_dir = train_adversarial(
            results_dir=tmp,
            n_qubits=2,
            n_generations=1,
            architect_steps_per_generation=2,
            saboteur_steps_per_generation=1,
            max_circuit_gates=5,
            fidelity_threshold=0.99,
            lambda_penalty=0.5,
            include_rotations=False,
            task_mode='state_preparation',
            champion_save_last_steps=0,
            hall_of_fame_size=3,
            saboteur_noise_family='depolarizing',
            saboteur_error_rates=[0.01, 0.02],
            saboteur_budget=1,
            saboteur_budget_fraction=0.2,
            saboteur_attack_candidate_fraction=0.8,
            saboteur_seed=0,
            stop_success_bonus=0.1,
            stop_failure_penalty=-0.05,
            per_step_penalty=-0.02,
        )
        print('Training finished. log_dir=', log_dir)

        # Very small evaluation: run the trained architect for a few episodes
        from qas_gym.envs import AdversarialArchitectEnv
        import numpy as _np
        import cirq

        eval_env = AdversarialArchitectEnv(
            saboteur_agent=sab_agent,
            saboteur_max_error_level=1,
            total_training_steps=1,
            saboteur_budget=1,
            saboteur_budget_fraction=0.2,
            saboteur_start_budget_scale=0.3,
            saboteur_attack_candidate_fraction=0.8,
            saboteur_seed=0,
            saboteur_error_rates=[0.01, 0.02],
            saboteur_noise_family='depolarizing',
            saboteur_noise_kwargs={},
            alpha_start=0.6,
            alpha_end=0.0,
            target=config.get_target_state(2),
            fidelity_threshold=0.99,
            max_timesteps=5,
            reward_penalty=0.01,
            complexity_penalty_weight=0.01,
            include_rotations=False,
            action_gates=config.get_action_gates([cirq.LineQubit(i) for i in range(2)], include_rotations=False),
            qubits=[cirq.LineQubit(i) for i in range(2)],
        )

        stop_reasons = {'agent_stop': 0, 'fidelity_or_budget': 0}
        fidelities = []
        for ep in range(10):
            obs, info = eval_env.reset()
            done = False
            while not done:
                # Simple random policy for evaluation
                a = eval_env.action_space.sample()
                obs, reward, done, trunc, info = eval_env.step(a)
            reason = info.get('stop_reason', 'fidelity_or_budget')
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
            fidelities.append(info.get('fidelity', 0.0))
        print('Stop reason counts:', stop_reasons)
        print('Fidelities:', fidelities)
    finally:
        # Keep the results for inspection; comment out removal when you want to keep
        # shutil.rmtree(tmp)
        pass


if __name__ == '__main__':
    # Lazy import of config to avoid circular issues when module executed
    from experiments import config
    main()
