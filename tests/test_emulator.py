
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.simulator import emulator_test
from qibolab.oneQ_emulator import create_oneQ_emulator
from qibolab.pulses import (
    PulseSequence,
)
from qibolab.sweeper import Parameter, QubitParameter, Sweeper

# pulse_simulator = emulator_platform.instruments['pulse_simulator']
# simulation_backend = pulse_simulator.simulation_backend

# HERE = pathlib.Path(__file__).parent

SWEPT_POINTS = 2
PLATFORM_NAMES = ["default_q0"]


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
    ro_pulse = platform.create_qubit_readout_pulse(0, 0)
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    sequence.add(platform.create_RX_pulse(0, 0))
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    if acquisition is AcquisitionType.DISCRIMINATION:
        result = platform.execute_pulse_sequence(sequence, options)
        assert result[0].samples.shape == (nshots,)
    else:
        with pytest.raises(TypeError) as excinfo:
            platform.execute_pulse_sequence(sequence, options)
        assert "Emulator does not support" in str(excinfo.value)


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
        assert "Sweep" in str(excinfo.value)


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
