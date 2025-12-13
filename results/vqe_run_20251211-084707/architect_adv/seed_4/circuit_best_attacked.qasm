// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(1), q(2), q(3)]
qreg q[3];


rz(pi*0.25) q[2];
cx q[0],q[2];
y q[2];
s q[0];
z q[2];
cx q[2],q[1];
