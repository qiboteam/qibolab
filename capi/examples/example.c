// This file is part of Qibolab
#include <stdio.h>
#include "qibolab/qibolab.h"

int main() {

  int* qasm_res = execute_qasm(
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

    printf("Samples:\n");
    for (int i = 0; i < 2*10; i++)
      printf("%d\n", qasm_res[i]);

    return 0;
}
