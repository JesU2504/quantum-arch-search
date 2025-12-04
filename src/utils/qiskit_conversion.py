import cirq
from qiskit import QuantumCircuit


def cirq_to_qiskit(circuit: cirq.Circuit) -> QuantumCircuit:
    """
    Convert a Cirq circuit to a Qiskit QuantumCircuit via QASM.

    Uses Cirq's QASM exporter to preserve qubit order and gate semantics, then
    relies on Qiskit's parser. Measurements are preserved if present.
    """
    qasm = cirq.qasm(circuit)
    return QuantumCircuit.from_qasm_str(qasm)


def ensure_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Ensure a circuit has measurements on all qubits.

    If no measurement is present, append measure_all to capture outcome stats.
    """
    if any(inst.operation.name == "measure" for inst in qc.data):
        return qc
    measured = qc.copy()
    measured.measure_all()
    return measured
