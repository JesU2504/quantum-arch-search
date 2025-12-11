"""
Stub for qiskit.primitives.containers to satisfy TorchQuantum imports on qiskit<1.0.
Provides a minimal PubResult placeholder.
"""

class PubResult:  # pragma: no cover - defensive shim
    def __init__(self, *args, **kwargs):
        # Mimic an object with a data dict attribute if needed
        self.data = {}
