// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


rz(pi*0.25) q[3];
y q[2];
s q[1];
t q[3];
cx q[0],q[3];
cx q[3],q[1];
cx q[2],q[3];
