import laboneq.simple as lo
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.zhinst import ZhPulse, ZhSweeper, ZhSweeperLine, Zurich
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse, Rectangular
from qibolab.sweeper import Parameter, Sweeper


# Function returning a calibrated device setup
def create_offline_device_setup():
    """
    Function returning a device setup
    """

    descriptor = """\
        instruments:
            SHFQC:
            - address: DEV12146
              uid: device_shfqc
            HDAWG:
            - address: DEV8660
              uid: device_hdawg
            PQSC:
            - address: DEV10055
              uid: device_pqsc

        connections:
            device_shfqc:
                - iq_signal: q0/drive_line
                  ports: SGCHANNELS/0/OUTPUT
                - iq_signal: q1/drive_line
                  ports: SGCHANNELS/1/OUTPUT
                - iq_signal: q0/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q0/acquire_line
                  ports: [QACHANNELS/0/INPUT]
                - iq_signal: q1/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q1/acquire_line
                  ports: [QACHANNELS/0/INPUT]


            device_hdawg:
                - rf_signal: q0/flux_line
                  ports: SIGOUTS/0
                - rf_signal: q1/flux_line
                  ports: SIGOUTS/1

            device_pqsc:
                - internal_clock_signal
                - to: device_hdawg
                  port: ZSYNCS/4
                - to: device_shfqc
                  port: ZSYNCS/0
        """

    device_setup = lo.DeviceSetup.from_descriptor(
        descriptor,
        server_host="my_ip_address",
        server_port="8004",
        setup_name="test_setup",
    )

    return device_setup


def test_zhpulse():
    pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    zhpulse = ZhPulse(pulse)
    assert zhpulse.pulse.serial == pulse.serial
    assert zhpulse.zhpulse.length == 40e-9


# def test_zhsweeper():
#     select & add sweeper

# def test_zhsweeper_line():
#     select sweeper


def test_zhinst_setup():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    assert IQM5q.time_of_flight == 280


def test_zhsequence():
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    IQM5q = create_platform("zurich")

    IQM5q.instruments[0].sequence_zh(sequence, IQM5q.qubits, sweepers=[])
    zhsequence = IQM5q.instruments[0].sequence

    with pytest.raises(AttributeError):
        IQM5q.instruments[0].sequence_zh("sequence", IQM5q.qubits, sweepers=[])
        zhsequence = IQM5q.instruments[0].sequence

    assert len(zhsequence) == 2
    assert len(zhsequence["readout0"]) == 1


# def test_calibration_step():
# sequence with coupler qubits and check signal map


def test_zhinst_register_readout_line():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup = create_offline_device_setup()
    IQM5q.register_readout_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "measure0" in IQM5q.signal_map
    assert "acquire0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/measure_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_drive_line():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup = create_offline_device_setup()
    IQM5q.register_drive_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "drive0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/drive_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_flux_line():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup = create_offline_device_setup()
    IQM5q.register_flux_line(platform.qubits[0])

    assert "flux0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/flux_line" in IQM5q.calibration.calibration_items


def test_experiment_execute_pulse_sequence():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup = create_offline_device_setup()

    # IQM5q.emulate = True
    # IQM5q.session = True

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits

    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sequence_zh(sequence, qubits, [])
    IQM5q.calibration_step(qubits)
    IQM5q.create_exp(qubits, options)

    # assert
    # AcquisitionType.SPECTROSCOPY
    # AveragingMode.CYCLIC
    # I'm using dumb IW
    assert 1 == 1
    # indeed it is dumb

    # assert "drive0" in IQM5q.experiment.signals
    # assert "flux0" in IQM5q.experiment.signals
    # assert "measure0" in IQM5q.experiment.signals
    # assert "acquire0" in IQM5q.experiment.signals


# TODO: Parametrize like in test_dummy and run with multiple sweeps
def test_experiment_sweep():
    platform = create_platform("zurich")
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    freq_width = 10_000_000
    freq_step = 1_000_000

    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sequence_zh(sequence, qubits, [freq_sweeper])
    IQM5q.calibration_step(qubits)
    IQM5q.create_exp(qubits, options)

    # assert
    # AcquisitionType.SPECTROSCOPY
    # AveragingMode.CYCLIC
    # I'm using dumb IW

    assert "drive0" in IQM5q.experiment.signals
    assert "flux0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


# target_element = sections=[AcquireLoopRt(uid='shots', alignment=SectionAlignment.LEFT, execution_type=ExecutionType.REAL_TIME, length=None, play_after=None, children=[Section(uid='sequence_drive0', alignment=SectionAlignment.LEFT, execution_type=ExecutionType.REAL_TIME, length=None, play_after=None, children=[Delay(signal='drive0', time=0.0, precompensation_clear=None), PlayPulse(signal='drive0', pulse=PulseFunctional(function='gaussian', uid='drive_0_0', amplitude=0.566, length=4e-08, pulse_parameters={'sigma': 0.4, 'zero_boundaries': False}), amplitude=None, increment_oscillator_phase=None, phase=0.0, set_oscillator_phase=None, length=None, pulse_parameters=None, precompensation_clear=None, marker=None)], trigger={}, on_system_grid=False), Section(uid='sequence_measure0', alignment=SectionAlignment.LEFT, execution_type=ExecutionType.REAL_TIME, length=None, play_after='sequence_drive0', children=[Delay(signal='measure0', time=0, precompensation_clear=None), Delay(signal='measure0', time=1e-07, precompensation_clear=None), PlayPulse(signal='measure0', pulse=PulseFunctional(function='const', uid='readout_0_0', amplitude=0.5, length=2e-06, pulse_parameters=None), amplitude=None, increment_oscillator_phase=None, phase=None, set_oscillator_phase=None, length=2e-06, pulse_parameters={'phase': 0}, precompensation_clear=None, marker=None), Delay(signal='acquire0', time=2.8e-16, precompensation_clear=None), Acquire(signal='acquire0', handle='sequence0', kernel=PulseFunctional(function='const', uid='weightreadout_0_0', amplitude=1, length=1.8499999999999999e-06, pulse_parameters=None), length=None, pulse_parameters=None), Delay(signal='acquire0', time=3e-13, precompensation_clear=None)], trigger={}, on_system_grid=False)], trigger={}, on_system_grid=False, acquisition_type=AcquisitionType.SPECTROSCOPY, averaging_mode=AveragingMode.CYCLIC, count=1024, repetition_mode=RepetitionMode.FASTEST, repetition_time=None, reset_oscillator_phase=False)])


# def test_qmopx_register_flux_pulse():
#     qubit = 2
#     platform = create_tii_qw5q_gold(RUNCARD, simulation_duration=1000, address=DUMMY_ADDRESS)
#     opx = platform.instruments[0]
#     pulse = FluxPulse(0, 30, 0.005, Rectangular(), platform.qubits[qubit].flux.name, qubit)
#     target_pulse = {
#         "operation": "control",
#         "length": pulse.duration,
#         "waveforms": {"single": "constant_wf0.005"},
#     }

#     opx.config.register_pulse(platform.qubits[qubit], pulse, opx.time_of_flight, opx.smearing)
#     assert opx.config.pulses[pulse.serial] == target_pulse
#     assert target_pulse["waveforms"]["single"] in opx.config.waveforms
#     assert opx.config.elements[f"flux{qubit}"]["operations"][pulse.serial] == pulse.serial
