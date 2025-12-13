// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1)]
qreg q[2];


s q[0];
y q[1];
ry(pi*0.25) q[0];
z q[0];
cx q[0],q[1];
