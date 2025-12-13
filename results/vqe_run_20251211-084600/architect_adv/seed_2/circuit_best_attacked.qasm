// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


cx q[1],q[0];
x q[0];
x q[0];
cx q[0],q[2];
y q[2];
cx q[1],q[0];
cx q[0],q[1];
cx q[3],q[0];
rz(pi*0.25) q[1];
y q[3];
