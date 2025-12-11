// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


s q[2];
x q[3];
t q[0];
cx q[1],q[2];
t q[3];
s q[2];
cx q[3],q[2];
