import os
import pathlib

import numpy as np
import pytest
from qutip import Options, identity, tensor

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.emulator.engines.generic import op_from_instruction
from qibolab.instruments.emulator.engines.qutip_engine import (
    QutipSimulator,
    extend_op_dim,
    function_from_array,
)
from qibolab.instruments.emulator.models import (
    general_no_coupler_model,
    models_template,
)
from qibolab.instruments.emulator.pulse_simulator import AVAILABLE_SWEEP_PARAMETERS
from qibolab.platform.load import PLATFORMS
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, QubitParameter, Sweeper

os.environ[PLATFORMS] = str(pathlib.Path(__file__).parent / "emulators/")

SWEPT_POINTS = 2
EMULATORS = ["default_q0"]
MODELS = [models_template, general_no_coupler_model]


@pytest.mark.parametrize("emulator", EMULATORS)
def test_emulator_initialization(emulators, emulator):
    platform = create_platform(emulator)
    platform.connect()
    platform.disconnect()


@pytest.mark.parametrize("emulator", EMULATORS)
@pytest.mark.parametrize(
    "acquisition",
    [AcquisitionType.DISCRIMINATION, AcquisitionType.INTEGRATION, AcquisitionType.RAW],
)
def test_emulator_execute_compute_overlaps(emulators, emulator, acquisition):
    nshots = 10  # 100
    platform = create_platform(emulator)
    pulse_simulator = platform.instruments["pulse_simulator"]
    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(0, 0))
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    if (
        acquisition is AcquisitionType.DISCRIMINATION
        or acquisition is AcquisitionType.INTEGRATION
    ):
        results = platform.execute([sequence], options)
        simulated_states = results["simulation"]["output_states"]
        overlaps = pulse_simulator.simulation_engine.compute_overlaps(simulated_states)
        if acquisition is AcquisitionType.DISCRIMINATION:
            assert results[0][0].samples.shape == (nshots,)
        else:
            assert results[0][0].voltage.shape == (nshots,)
    else:
        with pytest.raises(ValueError) as excinfo:
            platform.execute(sequence, options)
        assert "Current emulator does not support requested AcquisitionType" in str(
            excinfo.value
        )


@pytest.mark.parametrize("emulator", EMULATORS)
def test_emulator_execute_fast_reset(emulators, emulator):
    platform = create_platform(emulator)
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(
        nshots=None, fast_reset=True
    )  # fast_reset does nothing in emulator
    result = platform.execute([sequence], options)


@pytest.mark.parametrize("emulator", EMULATORS)
@pytest.mark.parametrize("fast_reset", [True, False])
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_single_sweep(
    emulators, emulator, fast_reset, parameter, average, acquisition, nshots
):
    platform = create_platform(emulator)
    sequence = PulseSequence()
    pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(SWEPT_POINTS)
    else:
        parameter_range = np.random.randint(1, 4, size=SWEPT_POINTS)
    sequence.add(pulse)
    if parameter in QubitParameter:
        sweeper = Sweeper(parameter, parameter_range, qubits=[platform.qubits[0]])
    else:
        sweeper = Sweeper(parameter, parameter_range, pulses=[pulse])

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
        fast_reset=fast_reset,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    if parameter in AVAILABLE_SWEEP_PARAMETERS:
        results = platform.execute([sequence], options, sweeper)

        assert pulse.serial and pulse.qubit in results
        if average:
            results_shape = results[pulse.qubit][0].statistical_frequency.shape
        else:
            results_shape = results[pulse.qubit][0].samples.shape
        assert results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)
    else:
        with pytest.raises(NotImplementedError) as excinfo:
            platform.execute([sequence], options, sweeper)
        assert "Sweep parameter requested not available" in str(excinfo.value)


@pytest.mark.parametrize("emulator", EMULATORS)
@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_double_sweep_false_history(
    emulators, emulator, parameter1, parameter2, average, acquisition, nshots
):
    platform = create_platform(emulator)
    pulse_simulator = platform.instruments["pulse_simulator"]
    pulse_simulator.output_state_history = False
    sequence = PulseSequence()
    pulse = platform.create_qubit_drive_pulse(qubit=0, start=0, duration=2)
    ro_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pulse.finish)
    sequence.add(pulse)
    sequence.add(ro_pulse)
    parameter_range_1 = (
        np.random.rand(SWEPT_POINTS)
        if parameter1 is Parameter.amplitude
        else np.random.randint(1, 4, size=SWEPT_POINTS)
    )
    parameter_range_2 = (
        np.random.rand(SWEPT_POINTS)
        if parameter2 is Parameter.amplitude
        else np.random.randint(1, 4, size=SWEPT_POINTS)
    )
    if parameter1 in QubitParameter:
        sweeper1 = Sweeper(parameter1, parameter_range_1, qubits=[platform.qubits[0]])
    else:
        sweeper1 = Sweeper(parameter1, parameter_range_1, pulses=[ro_pulse])
    if parameter2 in QubitParameter:
        sweeper2 = Sweeper(parameter2, parameter_range_2, qubits=[platform.qubits[0]])
    else:
        sweeper2 = Sweeper(parameter2, parameter_range_2, pulses=[pulse])

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT

    if (
        parameter1 in AVAILABLE_SWEEP_PARAMETERS
        and parameter2 in AVAILABLE_SWEEP_PARAMETERS
    ):
        results = platform.execute([sequence], options, sweeper1, sweeper2)

        assert ro_pulse.serial and ro_pulse.qubit in results

        if average:
            results_shape = results[pulse.qubit][0].statistical_frequency.shape
        else:
            results_shape = results[pulse.qubit][0].samples.shape

        assert (
            results_shape == (SWEPT_POINTS, SWEPT_POINTS)
            if average
            else (nshots, SWEPT_POINTS, SWEPT_POINTS)
        )


@pytest.mark.parametrize("emulator", EMULATORS)
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_single_sweep_multiplex(
    emulators, emulator, parameter, average, acquisition, nshots
):
    platform = create_platform(emulator)
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in platform.qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit=qubit, start=0)
        sequence.add(ro_pulses[qubit])
    parameter_range = (
        np.random.rand(SWEPT_POINTS)
        if parameter is Parameter.amplitude
        else np.random.randint(1, 4, size=SWEPT_POINTS)
    )

    if parameter in QubitParameter:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            qubits=[platform.qubits[qubit] for qubit in platform.qubits],
        )
    else:
        sweeper1 = Sweeper(
            parameter,
            parameter_range,
            pulses=[ro_pulses[qubit] for qubit in platform.qubits],
        )

    options = ExecutionParameters(
        nshots=nshots,
        averaging_mode=average,
        acquisition_type=acquisition,
    )
    average = not options.averaging_mode is AveragingMode.SINGLESHOT
    if parameter in AVAILABLE_SWEEP_PARAMETERS:
        results = platform.execute([sequence], options, sweeper1)

        for ro_pulse in ro_pulses.values():
            assert ro_pulse.serial and ro_pulse.qubit in results
            if average:
                results_shape = results[ro_pulse.qubit][0].statistical_frequency.shape
            else:
                results_shape = results[ro_pulse.qubit][0].samples.shape
            assert (
                results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)
            )


# pulse_simulator
def test_pulse_simulator_initialization(emulators):
    emulator = "default_q0"
    platform = create_platform(emulator)
    sim_opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
    pulse_simulator = platform.instruments["pulse_simulator"]
    pulse_simulator.connect()
    pulse_simulator.disconnect()
    pulse_simulator.dump()


def test_pulse_simulator_play_no_dissipation_dt_units_false_history_ro_exception(
    emulators,
):
    emulator = "default_q0"
    platform = create_platform(emulator)
    pulse_simulator = platform.instruments["pulse_simulator"]
    pulse_simulator.readout_error = {1: [0.1, 0.1]}
    pulse_simulator.runcard_duration_in_dt_units = True
    pulse_simulator.simulate_dissipation = False
    pulse_simulator.output_state_history = False
    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(0, 0))
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    execution_parameters = ExecutionParameters(nshots=10)
    with pytest.raises(ValueError) as excinfo:
        pulse_simulator.play({0: 0}, {}, sequence, execution_parameters)
    assert "Not all readout qubits are present in readout_error!" in str(excinfo.value)


# models.methods
def test_op_from_instruction():
    model = models_template
    model_config = model.generate_model_config()
    test_inst = model_config["drift"]["one_body"][1]
    test_inst2 = model_config["drift"]["two_body"][0]
    test_inst3 = (1.0, "b_2 ^ b_1 ^ b_0", ["2", "1", "0"])
    op_from_instruction(test_inst, multiply_coeff=False)
    op_from_instruction(test_inst2, multiply_coeff=False)
    op_from_instruction(test_inst3, multiply_coeff=False)


# engines.qutip_engine
@pytest.mark.parametrize("model", MODELS)
def test_update_sim_opts(model):
    model_config = model.generate_model_config()
    simulation_engine = QutipSimulator(model_config)
    sim_opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))


@pytest.mark.parametrize("model", MODELS)
def test_make_arbitrary_state(model):
    model_config = model.generate_model_config()
    simulation_engine = QutipSimulator(model_config)
    zerostate = simulation_engine.psi0.copy()
    dim = zerostate.shape[0]
    qibo_statevector = np.zeros(dim)
    qibo_statevector[2] = 1
    qibo_statevector = np.array(qibo_statevector.tolist())
    qibo_statedm = np.kron(
        qibo_statevector.reshape([dim, 1]), qibo_statevector.reshape([1, dim])
    )
    teststate = simulation_engine.make_arbitrary_state(
        qibo_statevector, is_qibo_state_vector=True
    )
    teststatedm = simulation_engine.make_arbitrary_state(
        qibo_statedm, is_qibo_state_vector=True
    )


@pytest.mark.parametrize("model", MODELS)
def test_state_from_basis_vector_exception(model):
    model_config = model.generate_model_config()
    simulation_engine = QutipSimulator(model_config)
    basis_vector0 = [0 for i in range(simulation_engine.nqubits)]
    cbasis_vector0 = [0 for i in range(simulation_engine.ncouplers)]
    simulation_engine.state_from_basis_vector(basis_vector0, None)
    combined_vector_list = [
        [basis_vector0 + [0], cbasis_vector0, "basis_vector"],
        [basis_vector0, cbasis_vector0 + [0], "cbasis_vector"],
    ]
    for combined_vector in combined_vector_list:
        with pytest.raises(Exception) as excinfo:
            basis_vector, cbasis_vector, error_vector = combined_vector
            simulation_engine.state_from_basis_vector(basis_vector, cbasis_vector)
        assert f"length of {error_vector} does not match" in str(excinfo.value)


def test_function_from_array_exception():
    y = np.ones([2, 2])
    x = np.ones([3, 2])
    with pytest.raises(ValueError) as excinfo:
        function_from_array(y, x)
    assert "y and x must have the same" in str(excinfo.value)


def test_extend_op_dim_exceptions():
    I2 = identity(2)
    I4 = identity(4)
    op_qobj = tensor(I2, I4)

    index_list1 = [[0], [0, 1], [2, 3], [4, 5, 6]]
    index_list2 = [[0], [4], [2, 3], [4, 5]]
    index_list3 = [[0], [0], [2, 3], [5, 4]]
    index_lists = [index_list1, index_list2, index_list3]

    for index_list in index_lists:
        with pytest.raises(Exception) as excinfo:
            extend_op_dim(op_qobj, *index_list)
        assert "mismatch" in str(excinfo.value)
