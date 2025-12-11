// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(2), q(3)]
qreg q[3];


z q[1];
ry(pi*0.25) q[0];
rz(pi*0.25) q[0];
y q[1];
cx q[1],q[2];
