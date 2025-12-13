// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7)]
qreg q[8];


cx q[2],q[7];
cx q[6],q[4];
y q[3];
rz(pi*0.25) q[0];
cx q[1],q[6];
cx q[2],q[5];
x q[7];
s q[3];
