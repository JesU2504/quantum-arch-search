# Experimental Plan: Adversarial Co‑Evolution vs. SOTA Baselines

Research goal
-------------
To rigorously demonstrate that Adversarial Co‑Evolution acts as a parameter‑free, dynamic regularizer that outperforms “Static Penalty” QAS methods in stability, robustness, and Pareto efficiency.


## Part 1 — “Brittleness” (Hyperparameter sensitivity)

Critique addressed: Prove that baselines fail even when “well‑tuned”.

- Context: Standard QAS uses $R = F - \lambda C$, where $C$ is cost (gates).
- Hypothesis: The “optimal” $\lambda$ is narrow and problem‑dependent. Outside this window, QAS fails. Adversarial works without tuning $\lambda$.

### Experiment 1.1 — Lambda sweep

- Task: 4‑qubit GHZ state preparation.
- Baselines: Train ArchitectEnv with static penalty $\lambda \in [0.001, 0.005, 0.01, 0.05, 0.1]$.
- Adversarial: Train AdversarialArchitectEnv (no fixed penalty).
- Metrics:
	- Success rate: % of seeds (out of 5) reaching fidelity > 0.99
	- Convergence variance: std. dev. of final gate counts
- Expected:
	- Low $\lambda$: High success, but high depth (bloat)
	- High $\lambda$: Low success (collapse)
	- Adversarial: High success, minimal depth
- Plot: Dual Y‑axis. X = $\lambda$. Left Y = Success rate. Right Y = Avg. depth. Add a horizontal line for Adversarial performance to show it beats the best‑tuned $\lambda$.


## Part 2 — Robustness to distribution shift (“Drift” test)

Critique addressed: Prove generalization to unseen noise.

- Context: A circuit trained on “Noise A” often fails on “Noise B”.
- Hypothesis: The Saboteur varies noise during training, creating an “ensemble robustness” effect.

### Experiment 2.1 — Cross‑noise evaluation

- Train:
	- Baseline: Trained on fixed depolarizing ($p=0.01$)
	- Robust: Trained against Saboteur
- Test sweep (evaluate both final circuits):
	- Coherent over‑rotation: $RX(\theta + \epsilon)$, $\epsilon \in [0, 0.1]$ (unseen by Saboteur)
	- Asymmetric noise: $p_x = 0.05,\ p_y=0.0,\ p_z=0.0$ (Saboteur usually uses symmetric)
- Metric: Fidelity retention ratio $F_{noisy} / F_{clean}$
- Expected: Robust circuit (shorter, attack‑exposed) decays slower on unseen error types.


## Part 3 — Scalability and the Pareto frontier

Critique addressed: Prove efficiency.

- Task: 5‑qubit GHZ (limit to 5 for sim feasibility).

### Experiment 3.1 — Pareto scatter

- Method:
	- Run 5 seeds of Static_Penalty (best $\lambda$ from Exp 1.1)
	- Run 5 seeds of Adversarial
- Metric: CNOT count vs fidelity
- Visualization: scatter plot (Adversarial points dominate top‑left: high F, low CNOT)


## Part 4 — Application: VQE on stretched H4

Critique addressed: Non‑trivial problem where entanglement matters.

- Task: Ground state energy of $H_4$ (linear chain) at $1.5\,\AA$.
- Why H4? $H_2$ is too easy (often depth‑1). $H_4$ requires correlation across 4 electrons.
- Setup:
	- Baselines: UCCSD (standard), HardwareEfficient (fixed)
	- Ours: Adversarial agent
- Metric: Energy error (Ha) vs CNOT count
- Win condition: Achieve chemical accuracy (1.6 mHa) with fewer CNOTs than UCCSD


## Part 5 — Computational overhead (HPC metric)

Critique addressed: Is the adversarial game worth the compute cost?

- Context: Adversarial training effectively doubles agent count. Question: time‑to‑solution (TTS)?

### Experiment 5.1 — Wall‑clock efficiency

- Method: Measure total training time (wall‑clock) to reach fidelity 0.99.
- Comparison:
	- Baseline (static penalty): fast per step; may need many episodes or restarts
	- Adversarial: slower per step (2 agents), but potentially fewer episodes to find robust structure
- Metric: Time‑to‑solution (seconds)
- Expected: Even if ~2× slower/step, Adversarial avoids the manual hyperparameter tuning “retry loop” and accelerates the overall research workflow.


## Part 6 — Defense against “Why not QEC?”

Chair’s challenge: “Why optimize physical circuits when QEC will solve noise eventually?”

Rebuttal:
- Resource reality: QEC needs >1000 physical qubits per logical qubit; we have <100 — our method works now.
- Logical design: Even with QEC, optimized logical circuits are needed; our framework transfers.
- Depth is king: QEC can’t fix infinite depth; minimizing depth reduces QEC burden.

### Experiment 6.1 — “NISQ vs QEC” resource plot

- Goal: Show QEC is infeasible at these sizes on limited hardware.
- Method: Estimate physical qubits for a 4‑qubit GHZ via surface code ($d=3$): approx. $4\times17=68$ physical qubits.
- Compare to our method (4 physical qubits).
- Visualization: Bar chart “Qubit overhead”
	- Bar A: Our method — 4 qubits
	- Bar B: Surface code QEC — 68 qubits
- Caption: “Our method enables high‑fidelity applications on hardware orders of magnitude too small for QEC.”


## Part 7 — Verification & smoke testing (stage verification)

Goal: Ensure components work before expensive HPC runs.

### Stage 7.1 — Saboteur efficacy check

- Test: Load a perfect GHZ circuit. Let the Saboteur act for 1 step with max budget.
- Verify: Fidelity must drop significantly below 1.0.
- Why: If fidelity stays 1.0, noise injection is broken and the Architect trains against a phantom.

### Stage 7.2 — VQE physics check

- Test: Initialize VQEArchitectEnv with a dummy (identity) circuit.
- Verify: Energy matches Hartree–Fock (≈ −1.117 Ha for H2 at eq.) or a random‑state energy, but not the exact FCI ground state yet.
- Why: Validates Hamiltonian mapping and expectation computation.

### Stage 7.3 — Parallelism overhead check

- Test: Run 1000 steps with DummyVecEnv (serial) vs SubprocVecEnv (parallel, n_envs=4).
- Verify: Parallel FPS ≈ $3\times$ serial (accounting for overhead).
- Why: If parallel is slower, env serialization is too heavy (common with Cirq objects).


Implementation notes
--------------------

- Parallelization: Use SubprocVecEnv in Stable‑Baselines3 to utilize all CPU cores (essential for 5–10 seeds).
- Logging: Log `cnot_count` explicitly (not just `len(circuit)`); include it in env info, e.g., `{ 'cnot_count': ... }`.
- Simulation backend: For <10 qubits, memory isn’t the bottleneck; Python object creation is. Reuse circuit objects where possible; consider compilation for other backends later.