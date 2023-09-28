import pytest
from qibo import gates

from qibolab.transpilers.blocks import Block, BlockingError


def test_count_2q_gates():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1), gates.CZ(0, 1), gates.H(0)])
    assert block.count_2q_gates() == 2


def test_add_gate():
    block = Block(qubits=(0, 1), gates=[gates.H(0)], entangled=False)
    block.add_gate(gates.CZ(0, 1))
    assert block.entangled == True
    assert block.count_2q_gates() == 1


def test_add_gate_error():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    with pytest.raises(BlockingError):
        block.add_gate(gates.CZ(0, 2))


def test_fuse_blocks():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(0, 1), gates=[gates.H(0)], entangled=False)
    fused = fuse_blocks(block_1, block_2)
    assert fused.qubits == (0, 1)
    assert fused.entangled == True
    assert fused.count_2q_gates() == 1


def test_fuse_blocks_error():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(1, 2), gates=[gates.CZ(1, 2)])
    with pytest.raises(BlockingError):
        fused = fuse_blocks(block_1, block_2)


# circ = Circuit(3)
# circ.add(gates.CZ(0, 1))
# circ.add(gates.CZ(0, 1))
# circ.add(gates.CZ(1, 2))
