import numpy as np
import pytest
from qutip import Options, identity, tensor

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.simulator import emulator_test
from qibolab.instruments.simulator.backends.generic import (
    dec_to_basis_string,
    op_from_instruction,
    print_Hamiltonian,
)
from qibolab.instruments.simulator.backends.qutip_backend import (
    Qutip_Simulator,
    extend_op_dim,
    function_from_array,
)
from qibolab.instruments.simulator.models import (
    general_no_coupler_model,
    models_template,
)
from qibolab.instruments.simulator.models.methods import load_model_params
from qibolab.oneQ_emulator import create_oneQ_emulator
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, QubitParameter, Sweeper

SWEPT_POINTS = 2
PLATFORM_NAMES = ["default_q0"]
MODELS = [models_template, general_no_coupler_model]


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_emulator_initialization(name):
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
    platform.connect()
    platform.disconnect()


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize(
    "acquisition",
    [AcquisitionType.DISCRIMINATION, AcquisitionType.INTEGRATION, AcquisitionType.RAW],
)
def test_emulator_execute_pulse_sequence(name, acquisition):
    nshots = 10  # 100
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
    pulse_simulator = platform.instruments["pulse_simulator"]
    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(0, 0))
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    if acquisition is AcquisitionType.DISCRIMINATION:
        result = platform.execute_pulse_sequence(sequence, options)
        assert result[0].samples.shape == (nshots,)
    else:
        with pytest.raises(TypeError) as excinfo:
            platform.execute_pulse_sequence(sequence, options)
        assert "Emulator does not support" in str(excinfo.value)
    pulse_simulator.print_sim_details()
    pulse_simulator.simulation_backend.fidelity_history()


@pytest.mark.parametrize("name", PLATFORM_NAMES)
def test_emulator_execute_pulse_sequence_fast_reset(name):
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    options = ExecutionParameters(
        nshots=None, fast_reset=True
    )  # fast_reset does nothing in emulator
    result = platform.execute_pulse_sequence(sequence, options)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("fast_reset", [True, False])
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_single_sweep(
    name, fast_reset, parameter, average, acquisition, nshots
):
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
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
    if parameter in platform.instruments["pulse_simulator"].available_sweep_parameters:
        results = platform.sweep(sequence, options, sweeper)

        assert pulse.serial and pulse.qubit in results
        if average:
            results_shape = results[pulse.qubit].statistical_frequency.shape
        else:
            results_shape = results[pulse.qubit].samples.shape
        assert results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)
    else:
        with pytest.raises(NotImplementedError) as excinfo:
            platform.sweep(sequence, options, sweeper)
        assert "Sweep parameter requested not available" in str(excinfo.value)


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_double_sweep(
    name, parameter1, parameter2, average, acquisition, nshots
):
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
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
        parameter1 in platform.instruments["pulse_simulator"].available_sweep_parameters
        and parameter2
        in platform.instruments["pulse_simulator"].available_sweep_parameters
    ):
        results = platform.sweep(sequence, options, sweeper1, sweeper2)

        assert ro_pulse.serial and ro_pulse.qubit in results

        if average:
            results_shape = results[pulse.qubit].statistical_frequency.shape
        else:
            results_shape = results[pulse.qubit].samples.shape

        assert (
            results_shape == (SWEPT_POINTS, SWEPT_POINTS)
            if average
            else (nshots, SWEPT_POINTS, SWEPT_POINTS)
        )


@pytest.mark.parametrize("name", PLATFORM_NAMES)
@pytest.mark.parametrize("parameter", Parameter)
@pytest.mark.parametrize("average", [AveragingMode.SINGLESHOT, AveragingMode.CYCLIC])
@pytest.mark.parametrize("acquisition", [AcquisitionType.DISCRIMINATION])
@pytest.mark.parametrize("nshots", [10, 20])
def test_emulator_single_sweep_multiplex(name, parameter, average, acquisition, nshots):
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
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
    if parameter in platform.instruments["pulse_simulator"].available_sweep_parameters:
        results = platform.sweep(sequence, options, sweeper1)

        for ro_pulse in ro_pulses.values():
            assert ro_pulse.serial and ro_pulse.qubit in results
            if average:
                results_shape = results[ro_pulse.qubit].statistical_frequency.shape
            else:
                results_shape = results[ro_pulse.qubit].samples.shape
            assert (
                results_shape == (SWEPT_POINTS,) if average else (nshots, SWEPT_POINTS)
            )


# pulse_simulator
def test_pulse_simulator_initialization():
    name = "default_q0"
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
    sim_opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
    pulse_simulator = platform.instruments["pulse_simulator"]
    pulse_simulator.update_sim_opts(sim_opts)
    pulse_simulator.connect()
    pulse_simulator.setup()
    pulse_simulator.start()
    pulse_simulator.stop()
    pulse_simulator.disconnect()


def test_pulse_simulator_play_def_execparams_no_dissipation_dt_units_ro_exception():
    name = "default_q0"
    runcard_folder = f"{emulator_test.__path__[0]}/{name}"
    platform = create_oneQ_emulator(runcard_folder)
    pulse_simulator = platform.instruments["pulse_simulator"]
    pulse_simulator.model_config.update({"readout_error": {1: [0.1, 0.1]}})
    pulse_simulator.model_config.update({"runcard_duration_in_dt_units": True})
    pulse_simulator.model_config.update({"simulate_dissipation": False})
    pulse_simulator.model_config["drift"].update({"two_body": []})
    pulse_simulator.model_config["dissipation"].update({"t1": []})
    print_Hamiltonian(pulse_simulator.model_config)
    pulse_simulator.update()
    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(0, 0))
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    with pytest.raises(ValueError) as excinfo:
        pulse_simulator.play({0: 0}, {}, sequence)
    assert "not present in ro_error_dict" in str(excinfo.value)
    pulse_simulator.simulation_backend.fidelity_history(time_in_dt=True)


# models.methods
def test_load_model_params():
    model_params_folder = emulator_test.__path__[0]
    device_name = "ibmfakebelem_q01"
    model_params = f"{model_params_folder}/{device_name}/model_params.yml"
    load_model_params(model_params)


# backends.generic
def test_dec_to_basis_string():
    dec_to_basis_string(x=1, nlevels=[3, 2, 2])


@pytest.mark.parametrize("model", MODELS)
def test_print_Hamiltonian(model):
    model_config = model.generate_model_config()
    print_Hamiltonian(model_config)


def test_op_from_instruction():
    model = models_template
    model_config = model.generate_model_config()
    test_inst = model_config["drift"]["one_body"][1]
    test_inst2 = model_config["drift"]["two_body"][0]
    op_from_instruction(test_inst, multiply_coeff=False)
    op_from_instruction(test_inst2, multiply_coeff=False)


# backends.qutip_backend
@pytest.mark.parametrize("model", MODELS)
def test_update_sim_opts(model):
    model_config = model.generate_model_config()
    simulation_backend = Qutip_Simulator(model_config)
    sim_opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))


@pytest.mark.parametrize("model", MODELS)
def test_make_arbitrary_state(model):
    model_config = model.generate_model_config()
    simulation_backend = Qutip_Simulator(model_config)
    zerostate = simulation_backend.psi0.copy()
    dim = zerostate.shape[0]
    qibo_statevector = np.zeros(dim)
    qibo_statevector[2] = 1
    qibo_statevector = np.array(qibo_statevector.tolist())
    qibo_statedm = np.kron(
        qibo_statevector.reshape([dim, 1]), qibo_statevector.reshape([1, dim])
    )
    teststate = simulation_backend.make_arbitrary_state(
        qibo_statevector, is_qibo_state_vector=True
    )
    teststatedm = simulation_backend.make_arbitrary_state(
        qibo_statedm, is_qibo_state_vector=True
    )


@pytest.mark.parametrize("model", MODELS)
def test_state_from_basis_vector_exception(model):
    model_config = model.generate_model_config()
    simulation_backend = Qutip_Simulator(model_config)
    basis_vector0 = [0 for i in range(simulation_backend.nqubits)]
    cbasis_vector0 = [0 for i in range(simulation_backend.ncouplers)]
    simulation_backend.state_from_basis_vector(basis_vector0, cbasis_vector0)
    combined_vector_list = [
        [basis_vector0 + [0], cbasis_vector0, "basis_vector"],
        [basis_vector0, cbasis_vector0 + [0], "cbasis_vector"],
    ]
    for combined_vector in combined_vector_list:
        with pytest.raises(Exception) as excinfo:
            basis_vector, cbasis_vector, error_vector = combined_vector
            simulation_backend.state_from_basis_vector(basis_vector, cbasis_vector)
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
