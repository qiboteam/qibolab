from qibo import Circuit, gates

from qibolab.transpilers.blocks import create_dag, initial_block_deomposition

circ = Circuit(6)
circ.add(gates.H(0))
circ.add(gates.H(1))
circ.add(gates.H(2))
circ.add(gates.CZ(0, 1))
circ.add(gates.CZ(0, 1))
circ.add(gates.CZ(2, 3))
circ.add(gates.CZ(1, 2))
circ.add(gates.CZ(0, 1))
circ.add(gates.CZ(2, 3))
circ.add(gates.H(0))
circ.add(gates.H(3))
circ.add(gates.H(4))
circ.add(gates.H(5))


dag = create_dag(circ)

blocks = initial_block_deomposition(circ)
for block in blocks:
    block.info()
