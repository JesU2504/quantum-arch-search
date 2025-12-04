#!/usr/bin/env python3
"""
Convenience entry point for adversarial co-evolution runs.
Delegates to experiments.adversarial.train_adversarial.
"""

from experiments.adversarial.train_adversarial import main

if __name__ == "__main__":
    main()
