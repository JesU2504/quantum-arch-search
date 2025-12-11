"""
Lightweight stub for qiskit_ibm_runtime to satisfy optional imports in torchquantum.

This project does not use IBM Runtime; the stub prevents hard import failures
when torchquantum tries to import QiskitRuntimeService on startup. Any actual
use will raise a clear error.
"""

class QiskitRuntimeService:  # pragma: no cover - defensive stub
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "qiskit_ibm_runtime is not installed. Install a compatible version if you need IBM Runtime."
        )


# TorchQuantum may import SamplerV2; provide a stub that errors when used.
class SamplerV2:  # pragma: no cover - defensive stub
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "qiskit_ibm_runtime.SamplerV2 is not available in this environment."
        )
