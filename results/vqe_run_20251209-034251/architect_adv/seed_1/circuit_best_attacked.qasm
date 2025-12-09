// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


cx q[3],q[0];
rx(pi*1.0) q[1];
cx q[1],q[2];
rz(0) q[0];
rz(0) q[1];
rz(pi*0.25) q[0];
cx q[1],q[3];
cx q[2],q[1];
