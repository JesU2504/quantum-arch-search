// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(3), q(4), q(5), q(6), q(7)]
qreg q[5];


s q[2];
y q[4];
x q[0];
cx q[3],q[4];
cx q[1],q[2];
s q[4];
