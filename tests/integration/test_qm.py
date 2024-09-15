import pytest

pytest.importorskip("qm")
pytest.importorskip("betterproto")
from qm import qua
from qm.qua import (
    Cast,
    align,
    declare,
    declare_stream,
    dual_demod,
    elif_,
    else_,
    fixed,
    for_,
    for_each_,
    if_,
    measure,
    play,
    save,
    set_dc_offset,
    stream_processing,
    update_frequency,
    wait,
)

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.qm.config import operation
from qibolab.sweeper import Parameter, Sweeper

from ..instruments.test_qm import QuaDummyBuilder, assert_ast_nodes


def test_qubit_flux(dummy_qrc):
    platform = create_platform("qm")

    natives = platform.natives.single_qubit
    sequence = (natives[0].RX() + natives[1].RX()) | (natives[0].MZ() + natives[1].MZ())

    f0 = platform.config("0/drive_lo").frequency + 1e8
    sweeper_freq0 = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 2e8, f0 + 2e8, 5e7),
        channels=[platform.qubits[0].drive],
    )
    f0 = platform.config("12/drive_lo").frequency - 1e8
    sweeper_freq1 = Sweeper(
        parameter=Parameter.frequency,
        range=(5.7e9 - 2e8, 5.7e9 + 2e8, 5e7),
        channels=[platform.qubits[1].drive],
    )
    sweeper_bias = Sweeper(
        parameter=Parameter.offset,
        range=(-0.2, 0.2, 0.1),
        channels=[platform.qubits[0].flux, platform.qubits[1].flux],
    )

    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=1000,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    result = platform.execute(
        [sequence], options, [[sweeper_bias], [sweeper_freq0, sweeper_freq1]]
    )
    experiment = result["program"]

    # write target QUA program for the above experiment
    qd0 = operation(sequence[0][1])
    qd1 = operation(sequence[1][1])
    ro0 = operation(sequence.acquisitions[0][1])
    ro1 = operation(sequence.acquisitions[1][1])
    with qua.program() as target_experiment:
        v1 = declare(
            int,
        )
        v2 = declare(
            fixed,
        )
        v3 = declare(
            fixed,
        )
        v4 = declare(
            fixed,
        )
        v5 = declare(
            fixed,
        )
        v6 = declare(
            fixed,
        )
        v7 = declare(
            int,
        )
        v8 = declare(
            int,
        )
        a1 = declare(
            int,
            value=[
                -100000000,
                -50000000,
                0,
                50000000,
                100000000,
                150000000,
                200000000,
                250000000,
            ],
        )
        a2 = declare(
            int,
            value=[
                -200000000,
                -150000000,
                -100000000,
                -50000000,
                0,
                50000000,
                100000000,
                150000000,
            ],
        )
        wait((4 + (0 * (Cast.to_int(v2) + Cast.to_int(v3)))), "0/acquisition")
        wait((4 + (0 * (Cast.to_int(v4) + Cast.to_int(v5)))), "1/acquisition")
        with for_(v1, 0, (v1 < 1000), (v1 + 1)):
            with for_(v6, -0.2, (v6 < 0.15000000000000002), (v6 + 0.1)):
                with if_(v6 >= 0.5):
                    set_dc_offset("0/flux", "single", 0.5)
                with elif_(v6 <= -0.5):
                    set_dc_offset("0/flux", "single", -0.5)
                with else_():
                    set_dc_offset("0/flux", "single", v6)
                with if_(v6 >= 0.5):
                    set_dc_offset("1/flux", "single", 0.5)
                with elif_(v6 <= -0.5):
                    set_dc_offset("1/flux", "single", -0.5)
                with else_():
                    set_dc_offset("1/flux", "single", v6)
                with for_each_((v7, v8), (a1, a2)):
                    update_frequency("0/drive", v7, "Hz", False)
                    update_frequency("1/drive", v8, "Hz", False)
                    align()
                    play(qd0, "0/drive")
                    play(qd1, "1/drive")
                    wait(11, "0/acquisition")
                    wait(11, "1/acquisition")
                    measure(
                        ro0,
                        "0/acquisition",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", v2),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", v3),
                    )
                    r1 = declare_stream()
                    save(v2, r1)
                    r2 = declare_stream()
                    save(v3, r2)
                    measure(
                        ro1,
                        "1/acquisition",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", v4),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", v5),
                    )
                    r3 = declare_stream()
                    save(v4, r3)
                    r4 = declare_stream()
                    save(v5, r4)
                    wait(
                        250,
                    )
        with stream_processing():
            r1.buffer(8).buffer(4).average().save(f"{ro0}_0/acquisition_I")
            r2.buffer(8).buffer(4).average().save(f"{ro0}_0/acquisition_Q")
            r3.buffer(8).buffer(4).average().save(f"{ro1}_1/acquisition_I")
            r4.buffer(8).buffer(4).average().save(f"{ro1}_1/acquisition_Q")

    # compare experiment generated by the driver with the target
    build = experiment.build(QuaDummyBuilder())
    target_build = target_experiment.build(QuaDummyBuilder())
    assert_ast_nodes(build.script, target_build.script)
    assert_ast_nodes(build.result_analysis, target_build.result_analysis)
