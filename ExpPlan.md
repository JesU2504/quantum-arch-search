# Experimental Plan: Adversarial Co‑Evolution vs. SOTA Baselines

## Research goal
To rigorously demonstrate that Adversarial Co‑Evolution acts as a parameter‑free, dynamic regularizer that outperforms “Static Penalty” QAS methods in stability, robustness, and Pareto efficiency.

---

## Reviewer-motivated Expansions and Clarifications

- **Motivation and Target Audience**
    - The brittleness and hyperparameter sensitivity of quantum architecture search (QAS) *is* a widely acknowledged challenge. However, prior work (including grid search, Bayesian optimization, cross-validation, or ensemble methods) already attempts to mitigate these issues. Our work aims to show *qualitatively new robustness* enabled by adversarial regularization, and not simply an alternative to hyperparameter tuning.
    - For conference-level scrutiny, we emphasize not only simulated benchmarks but also the approach's *practical relevance*, addressing both resource-constrained (near-term) and scalable (future-proof) quantum regimes.

- **Scope and Limitation Notices**
    - **Simulation Boundaries**: Our core experiments employ 3–5 qubit instances due to classical simulation limits. Where possible, we will report *scaling trends* and outline extrapolation challenges.
    - **Real Hardware Transfer**: While we do not run on quantum hardware in this version, we explicitly compare noise models, and plan to release open-source protocols for future device-level testing.
    - **Baselines**: We include strong baselines, including best-practice hyperparameter selection, and highlight limitations of automated tuning because our adversarial method does not require post hoc parameter adjustment.

---

## Part 1 — “Brittleness” (Hyperparameter sensitivity)
Critique addressed: Prove that baselines fail even when “well‑tuned”.

- Context: Standard QAS uses $R = F - \lambda C$, where $C$ is cost (gates).
- **Hypothesis**: The “optimal” $\lambda$ is narrow and problem‑dependent. Outside this window, QAS fails. Adversarial works without tuning $\lambda$, and yields solutions robust to variations in architectural constraints.
- **Addition**: We will compare not only fixed $\lambda$, but also *best-practice automated tuning* (grid/Bayesian search, ensemble methods) to ensure our gains are not simply due to lack of parameter optimization in the baseline.

### Experiment 1.1 — Lambda sweep
- Task: 4‑qubit GHZ state preparation.
- Baselines: Train ArchitectEnv with static penalty $\lambda \in [0.001, 0.005, 0.01, 0.05, 0.1]$, plus one automatically tuned $\lambda$ per best practice.
- Adversarial: Train AdversarialArchitectEnv (no fixed penalty).
- Metrics:
    - Success rate: % of seeds (out of 5) reaching fidelity > 0.99
    - Convergence variance: std. dev. of final gate counts
- **Additional metric**: Number of manual/automated hyperparameter search runs required by static baseline.
- Expected:
    - Low $\lambda$: High success, but high depth (bloat)
    - High $\lambda$: Low success (collapse)
    - Adversarial: High success, minimal depth
- Plot: Dual Y‑axis. X = $\lambda$. Left Y = Success rate. Right Y = Avg. depth. Add a horizontal line for Adversarial performance to show it beats the best‑tuned $\lambda$.
- **Justification**: This demonstrates robustness *across* tuning regimes and justifies extra complexity of the adversarial approach.

---

## Part 2 — Robustness to distribution shift (“Drift” test)
Critique addressed: Prove generalization to unseen noise.

- Context: A circuit trained on “Noise A” may fail on “Noise B”.
- **Hypothesis**: The Saboteur generates training-time noise diversity, yielding “ensemble robustness.”
- **Expansion**: Real-world device noise can be time-varying or non-iid. Our tests include attacks outside the Saboteur’s training distribution to measure true generalization.

### Experiment 2.1 — Cross‑noise evaluation
- Train:
    - Baseline: Trained on fixed depolarizing ($p=0.01$)
    - Robust: Trained against Saboteur
- Test sweep (evaluate both final circuits):
    - Coherent over‑rotation: $RX(\theta + \epsilon)$, $\epsilon \in [0, 0.1]$ (unseen by Saboteur)
    - Asymmetric noise: $p_x = 0.05, p_y=0.0, p_z=0.0$
    - **[New] Realistic “device drift”: Sampled time-varying noise amplitude**
- Metric: Fidelity retention ratio $F_{noisy} / F_{clean}$
- **Addition**: If possible, a transfer test to a real-device-inspired noise channel.
- Expected: Robust circuit (shorter, attack‑exposed) decays slower on unseen error types.

---

## Part 3 — Scalability and the Pareto frontier
Critique addressed: Prove efficiency and scalability.

- Task: 5‑qubit GHZ (simulate up to feasible limit).
- **Expansion**: To the extent possible, also run or estimate for larger systems and report “scaling wall” (resource required as function of n-qubits).
- **Justification**: Real-world utility depends on scaling, so even indicative scaling data is critical.

### Experiment 3.1 — Pareto scatter
- Method:
    - Run 5 seeds of Static_Penalty (best $\lambda$ from Exp 1.1)
    - Run 5 seeds of Adversarial
- Metric: CNOT count vs fidelity
- Visualization: scatter plot (Adversarial points dominate top‑left: high F, low CNOT)

---

## Part 4 — Application: VQE on stretched H4
Critique addressed: Demonstrate value on non‑trivial, physically relevant problem.

- Task: Ground state energy of $H_4$ (linear chain) at $1.5\,\AA$.
- **Justification for H4**: $H_2$ is too easy. $H_4$ requires correlation across 4 electrons.
- Setup:
    - Baselines: UCCSD (standard), HardwareEfficient (fixed)
    - Ours: Adversarial agent
- Metric: Energy error (Ha) vs CNOT count; **[New] also wall-clock time and convergence rate**
- Win condition: Achieve chemical accuracy (1.6 mHa) with fewer CNOTs than UCCSD; show gain vs manual tuning.
- **Implementation Note**: This experiment is under active completion; results will be prioritized for conference submission.

---

## Part 5 — Computational overhead (HPC metric)
Critique addressed: Is the adversarial game worth the compute cost?

- Context: Adversarial training doubles agent count; must justify extra cost.
- **Expansion**: Compare total wall-clock time, number of hyperparameter search runs, convergence rate, and hardware resource requirements between Adversarial and Static approaches.
- **Addition**: Provide estimate of researcher labor (time to first valid solution, not just clock time).

### Experiment 5.1 — Wall‑clock efficiency
- Method: Measure total training time (wall‑clock) to reach fidelity 0.99 under both methods.
- Comparison:
    - Baseline (static penalty): fast per step; more trial-and-error cycles
    - Adversarial: slower per step; fewer retries
- Metric: Time‑to‑solution (seconds), and number of human/equipment cycles required
- Expected: Any computational penalty for Adversarial is offset by avoiding manual hyperparameter tuning retry loops.
- **Justification**: Demonstrates that real-world efficiency, not just theoretical quality, is improved.

---

## Part 6 — Defense against “Why not QEC?”
Chair’s challenge: “Why optimize physical circuits when QEC will solve noise eventually?”

**Rebuttal:**
- Resource reality: QEC needs >1000 physical qubits per logical; most devices have <100 — ours is relevant now.
- Logical design: Even in QEC era, optimized logical circuits are needed.
- Depth is king: QEC alone cannot reduce circuit depth.
- **[Addition] Quantitative: Compare to state-of-the-art noise mitigation techniques beyond simple error correction.**

### Experiment 6.1 — “NISQ vs QEC” resource plot
- Goal: Show QEC is infeasible for current hardware sizes.
- Method: Estimate physical qubits needed for 4‑qubit GHZ via surface code ($d=3$): approx. $4\times17=68$ qubits.
- Compare to our method (4 qubits).
- Visualization: Bar chart “Qubit overhead”
    - Bar A: Our method — 4 qubits
    - Bar B: Surface code QEC — 68 qubits
- Caption: Our method enables high‑fidelity applications on hardware much too small for QEC.
- **[Addition] Discussion of resource consumption vs other types of noise mitigation where appropriate.**

---

## Part 7 — Verification & smoke testing (stage verification)
Goal: Ensure components work before expensive HPC runs.

### Stage 7.1 — Saboteur efficacy check
- Test: Load a perfect GHZ circuit. Let Saboteur act for 1 step with max budget.
- Verify: Fidelity must drop significantly; if not, noise injection is faulty.

### Stage 7.2 — VQE physics check
- Test: Initialize VQEArchitectEnv with dummy circuit.
- Verify: Baseline energy matches Hartree–Fock or random-state, not FCI ground state.
- Why: Validates Hamiltonian mapping.

### Stage 7.3 — Parallelism overhead check
- Test: Run 1000 steps serial vs parallel.
- Verify: Parallel throughput ~3x serial (with overhead).
- Why: Detect serialization bottlenecks (common with Cirq).

---

## Implementation notes

- Parallelization: Use SubprocVecEnv in Stable‑Baselines3 to use all CPU cores.
- Logging: Log `cnot_count` explicitly in env info.
- Simulation backend: For <10 qubits, memory isn’t the bottleneck; reuse circuit objects; plan for future backend upgrades.

---

## Additional Conference Readiness Checklist

- [ ] All experiments reported above are automated and reproducible via provided scripts
- [ ] Scaling data (even if limited) for larger n-qubit benchmarks
- [ ] Honest discussion of all limitations (hardware, code, resource boundaries)
- [ ] All plots and data include best-practice-tuned baselines, not strawman comparisons
- [ ] Future work: direct inference speed, transferability to hardware

---
