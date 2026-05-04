"""Test compile with and without rel_phase sweeper"""

import numpy as np

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.q1asm.ast_ import Add, Line, SetPhDelta
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.instruments.qblox.sequence.sequence import compile
from qibolab._core.pulses import VirtualZ
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper


def test_virtualz_with_phase_sweeper():
    vz = VirtualZ(phase=0.25)
    seq = PulseSequence([("ch1", vz)])
    sweeper = Sweeper(
        parameter=Parameter.relative_phase,
        values=np.array([0.1, 0.2, 0.3]),
        pulses=[vz],
    )
    options = ExecutionParameters(nshots=5, relaxation_time=10)
    result = compile(seq, [[sweeper]], options, sampling_rate=1.0, merged_vzs=False)
    instrs = result["ch1"].program.elements

    line_instrs = [i for i in instrs if isinstance(i, Line)]
    add_instrs = [i for i in line_instrs if isinstance(i.instruction, Add)]
    assert add_instrs, "Expected at least one 'Add' instruction for phase sweep."
    assert any(
        isinstance(i.instruction, Add)
        and i.instruction.destination == Registers.phase_delta.value
        for i in add_instrs
    ), "Expected 'Add' instruction to use phase_delta register as argument."
    assert not any(isinstance(i.instruction, SetPhDelta) for i in line_instrs), (
        "Did not expect 'SetPhDelta' instruction when phase is swept."
    )


def test_virtualz_without_phase_sweeper():
    vz = VirtualZ(phase=0.25)
    seq = PulseSequence([("ch1", vz)])
    options = ExecutionParameters(nshots=1, relaxation_time=10)
    result = compile(seq, [], options, 1.0, merged_vzs=True)
    instrs = result["ch1"].program.elements

    line_instrs = [i for i in instrs if isinstance(i, Line)]
    assert any(isinstance(i.instruction, SetPhDelta) for i in line_instrs), (
        "Expected at least one 'SetPhDelta' instruction for static phase."
    )
