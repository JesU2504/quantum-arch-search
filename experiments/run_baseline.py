#!/usr/bin/env python3
"""
Convenience entry point for architect-only baseline runs.
Delegates to experiments.architect.train_architect.
"""

from experiments.architect.train_architect import main

if __name__ == "__main__":
    main()
