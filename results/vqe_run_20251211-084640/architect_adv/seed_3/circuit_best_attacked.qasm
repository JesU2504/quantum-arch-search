// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(7)]
qreg q[7];


s q[0];
cx q[2],q[6];
cx q[1],q[4];
t q[5];
cx q[0],q[2];
x q[6];
cx q[3],q[4];
s q[6];
cx q[6],q[3];
