// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


y q[2];
rz(pi*0.25) q[1];
y q[3];
cx q[0],q[1];
