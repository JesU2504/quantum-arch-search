# quantum-arch-search

Research code for co-evolutionary quantum circuit architecture search (Architect vs Saboteur).

Quickstart
---------
1. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the smoke test to verify utilities:

```bash
python3 tests/test_utils_smoke.py
```

3. Run experiments (example):

```bash
python3 run_experiments.py
```

Notes
-----
- This project uses Cirq for circuit simulation and Stable-Baselines3 (PPO) for RL agents.
- Circuits are stored as JSON using helper functions in `src/qas_gym/utils.py` (`save_circuit`/`load_circuit`).
- For reproducibility, we recommend setting RNG seeds (numpy, SB3, Cirq) before running long experiments.

