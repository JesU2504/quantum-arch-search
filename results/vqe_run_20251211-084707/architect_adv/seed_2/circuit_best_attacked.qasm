// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


cx q[1],q[3];
rz(pi*0.25) q[0];
s q[2];
x q[1];
z q[3];
cx q[0],q[1];
t q[3];
cx q[1],q[2];
z q[0];
y q[3];
y q[1];
