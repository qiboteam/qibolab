// This file is part of Qibolab
#include <stdio.h>
#include "qibolab/qibolab.h"

int main() {

    double* qasm_res = execute_qasm(
        "OPENQASM 2.0;" \
        "include \"qelib1.inc\";" \
        "qreg q[3];" \
        "creg a[2];" \
        "cx q[0],q[2];" \
        "x q[1];" \
        "swap q[0],q[1];" \
        "cx q[1],q[0];" \
        "measure q[0] -> a[0];" \
        "measure q[2] -> a[1];",
        "dummy", 10);

    printf("samples = %f", qasm_res[0]);

    return 0;
}
