// Generated from Cirq v1.6.1

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];


cx q[3],q[2];
rx(pi*-0.125) q[2];
cx q[3],q[1];
rx(pi*-0.0625) q[2];
rz(pi*-0.125) q[1];
cx q[3],q[2];
rz(pi*0.125) q[1];
cx q[0],q[2];
h q[3];
rz(pi*-0.125) q[0];
rx(pi*0.125) q[2];
rz(pi*0.0625) q[3];
rz(pi*0.125) q[0];
h q[3];
rz(pi*-0.0625) q[2];
cx q[0],q[3];
rx(pi*-0.125) q[0];
cx q[0],q[2];
cx q[2],q[3];
cx q[1],q[2];
rx(pi*0.125) q[2];
cx q[1],q[0];
h q[2];
rx(pi*0.125) q[0];
rx(pi*0.125) q[1];
rz(pi*-0.125) q[2];
rz(pi*0.125) q[0];
cx q[2],q[3];
rx(pi*-0.125) q[0];
rx(pi*-0.0625) q[2];
cx q[3],q[1];
rz(pi*0.0625) q[0];
cx q[1],q[2];
rz(pi*-0.125) q[3];
cx q[2],q[1];
rx(pi*-0.125) q[2];
