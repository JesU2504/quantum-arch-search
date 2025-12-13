// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1)]
qreg q[2];


s q[0];
s q[1];
rz(pi*0.25) q[1];
cx q[0],q[1];
y q[0];
