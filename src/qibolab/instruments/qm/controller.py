import shutil
import tempfile
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.octave import QmOctaveConfig
from qm.simulate.credentials import create_credentials
from qualang_tools.simulator_tools import create_simulator_controller_connections

from qibolab.components import AcquireChannel, Config, DcChannel, IqChannel
from qibolab.components.configs import IqConfig, OscillatorConfig
from qibolab.execution_parameters import ExecutionParameters
from qibolab.identifier import ChannelId
from qibolab.instruments.abstract import Controller
from qibolab.instruments.qm.components.configs import (
    OpxOutputConfig,
    QmAcquisitionConfig,
)
from qibolab.pulses import Acquisition, Align, Delay, Pulse, Readout
from qibolab.sequence import PulseSequence
from qibolab.sweeper import ParallelSweepers, Parameter, Sweeper
from qibolab.unrolling import Bounds, unroll_sequences

from .config import SAMPLING_RATE, QmConfig, operation
from .program import ExecutionArguments, create_acquisition, program
from .program.sweepers import check_frequency_bandwidth, sweeper_amplitude

OCTAVE_ADDRESS_OFFSET = 11000
"""Offset to be added to Octave addresses, because they must be 11xxx, where
xxx are the last three digits of the Octave IP address."""
CALIBRATION_DB = "calibration_db.json"
"""Name of the file where the mixer calibration is stored."""

__all__ = ["QmController", "Octave"]

BOUNDS = Bounds(
    waveforms=int(4e4),
    readout=30,
    instructions=int(1e6),
)


@dataclass(frozen=True)
class Octave:
    """User-facing object for defining Octave configuration."""

    name: str
    """Name of the device."""
    port: int
    """Network port of the Octave in the cluster configuration."""
    connectivity: str
    """OPXplus that acts as the waveform generator for the Octave."""


def declare_octaves(octaves, host, calibration_path=None):
    """Initiate Octave configuration and add octaves info.

    Args:
        octaves (dict): Dictionary containing :class:`qibolab.instruments.qm.devices.Octave` objects
            for each Octave device in the experiment configuration.
        host (str): IP of the Quantum Machines controller.
        calibration_path (str): Path to the JSON file with the mixer calibration.
    """
    if len(octaves) == 0:
        return None

    config = QmOctaveConfig()
    if calibration_path is not None:
        config.set_calibration_db(calibration_path)
    for octave in octaves.values():
        config.add_device_info(octave.name, host, OCTAVE_ADDRESS_OFFSET + octave.port)
    return config


def fetch_results(result, acquisitions):
    """Fetches results from an executed experiment.

    Args:
        result: Result of the executed experiment.
        acquisition: Dictionary containing :class:`qibolab.instruments.qm.acquisition.Acquisition` objects.

    Returns:
        Dictionary with the results in the format required by the platform.
    """
    handles = result.result_handles
    handles.wait_for_all_values()  # for async replace with ``handles.is_processing()``
    results = defaultdict(list)
    for acquisition in acquisitions:
        data = acquisition.fetch(handles)
        for serial, result in zip(acquisition.keys, data):
            results[serial].append(result)

    # collapse single element lists for back-compatibility
    return {
        key: value[0] if len(value) == 1 else value for key, value in results.items()
    }


def find_sweepers(
    sweepers: list[ParallelSweepers], parameter: Parameter
) -> list[Sweeper]:
    """Find sweepers of given parameter in order to register specific pulses.

    Duration and amplitude sweepers may require registering additional pulses
    in the QM ``config``.
    """
    return [s for ps in sweepers for s in ps if s.parameter is parameter]


class QmController(Controller):
    """:class:`qibolab.instruments.abstract.Controller` object for controlling
    a Quantum Machines cluster.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language.
    The ``config`` file is generated using the ``dataclass`` objects defined in
    :mod:`qibolab.instruments.qm.config`.
    The QUA program is generated using the methods in :mod:`qibolab.instruments.qm.program`.
    Controllers, elements and pulses are added in the ``config`` after a pulse sequence is given,
    so that only elements related to the participating channels are registered.
    """

    name: str
    """Name of the instrument instance."""
    address: str
    """IP address and port for connecting to the OPX instruments.

    Has the form XXX.XXX.XXX.XXX:XXX.
    """

    octaves: dict[str, Octave]
    """Dictionary containing the
    :class:`qibolab.instruments.qm.controller.Octave` instruments being
    used."""

    bounds: str = "qm/bounds"
    """Maximum bounds used for batching in sequence unrolling."""
    calibration_path: Optional[PathLike] = None
    """Path to the JSON file that contains the mixer calibration."""
    write_calibration: bool = False
    """Require writing permissions on calibration DB."""
    _calibration_path: Optional[PathLike] = None
    """The calibration path for internal use.

    Cf. :attr:`calibration_path` for its role. This might be set to a different one
    internally to avoid writing attempts over a file for which the user has only read
    access (because TinyDB, through QUA, is often attempting to open it in append mode).
    """
    script_file_name: Optional[str] = None
    """Name of the file that the QUA program will dumped in that after every
    execution.

    If ``None`` the program will not be dumped.
    """

    manager: Optional[QuantumMachinesManager] = None
    """Manager object used for controlling the Quantum Machines cluster."""
    is_connected: bool = False
    """Boolean that shows whether we are connected to the QM manager."""

    config: QmConfig = Field(default_factory=QmConfig)
    """Configuration dictionary required for pulse execution on the OPXs."""

    simulation_duration: Optional[int] = None
    """Duration for the simulation in ns.

    If given the simulator will be used instead of actual hardware
    execution.
    """
    cloud: bool = False
    """If ``True`` the QM cloud simulator is used which does not require access
    to physical instruments.

    This assumes that a proper cloud address has been given.
    If ``False`` and ``simulation_duration`` was given, then the built-in simulator
    of the instruments is used. This requires connection to instruments.
    Default is ``False``.
    """

    def model_post_init(self, __context):
        if self.simulation_duration is not None:
            # convert simulation duration from ns to clock cycles
            self.simulation_duration //= 4

    @property
    def sampling_rate(self):
        """Sampling rate of Quantum Machines instruments."""
        return SAMPLING_RATE

    def _temporary_calibration(self):
        if self.calibration_path is not None:
            if self.write_calibration:
                self._calibration_path = self.calibration_path
            else:
                self._calibration_path = tempfile.mkdtemp()
                shutil.copy(
                    Path(self.calibration_path) / CALIBRATION_DB,
                    Path(self._calibration_path) / CALIBRATION_DB,
                )

    def _reset_temporary_calibration(self):
        if self._calibration_path != self.calibration_path:
            assert self._calibration_path is not None
            shutil.rmtree(self._calibration_path)
            self._calibration_path = None

    def setup(self, *args, **kwargs):
        """Complying with abstract instrument interface.

        Not needed for this instrument.
        """

    def connect(self):
        """Connect to the Quantum Machines manager."""
        host, port = self.address.split(":")
        self._temporary_calibration()
        octave = declare_octaves(self.octaves, host, self._calibration_path)
        credentials = None
        if self.cloud:
            credentials = create_credentials()
        self.manager = QuantumMachinesManager(
            host=host, port=int(port), octave=octave, credentials=credentials
        )
        self.is_connected = True

    def disconnect(self):
        """Disconnect from QM manager."""
        self._reset_temporary_calibration()
        if self.manager is not None:
            self.manager.close_all_quantum_machines()
            self.manager.close()
            self.is_connected = False

    def configure_device(self, device: str):
        """Add device in the ``config``."""
        if "octave" in device:
            self.config.add_octave(device, self.octaves[device].connectivity)
        else:
            self.config.add_controller(device)

    def configure_channel(self, channel: ChannelId, configs: dict[str, Config]):
        """Add element (QM version of channel) in the config."""
        config = configs[channel]
        ch = self.channels[channel]
        self.configure_device(ch.device)

        if isinstance(ch, DcChannel):
            assert isinstance(config, OpxOutputConfig)
            self.config.configure_dc_line(channel, ch, config)

        elif isinstance(ch, IqChannel):
            assert ch.lo is not None
            assert isinstance(config, IqConfig)
            lo_config = configs[ch.lo]
            assert isinstance(lo_config, OscillatorConfig)
            self.config.configure_iq_line(channel, ch, config, lo_config)

        elif isinstance(ch, AcquireChannel):
            assert ch.probe is not None
            assert isinstance(config, QmAcquisitionConfig)
            probe = self.channels[ch.probe]
            probe_config = configs[ch.probe]
            assert isinstance(probe, IqChannel)
            assert isinstance(probe_config, IqConfig)
            assert probe.lo is not None
            lo_config = configs[probe.lo]
            assert isinstance(lo_config, OscillatorConfig)
            self.configure_device(ch.device)
            self.config.configure_acquire_line(
                channel, ch, probe, config, probe_config, lo_config
            )

        else:
            raise TypeError(f"Unknown channel type: {type(ch)}.")

    def configure_channels(self, configs: dict[str, Config], channels: set[ChannelId]):
        """Register channels in the sequence in the QM ``config``."""
        for id in channels:
            self.configure_channel(id, configs)

    def register_pulse(self, channel: ChannelId, pulse: Union[Pulse, Readout]) -> str:
        """Add pulse in the QM ``config``.

        And return corresponding operation.
        """
        ch = self.channels[channel]
        if isinstance(ch, DcChannel):
            assert isinstance(pulse, Pulse)
            return self.config.register_dc_pulse(channel, pulse)
        if isinstance(ch, IqChannel):
            assert isinstance(pulse, Pulse)
            return self.config.register_iq_pulse(channel, pulse)
        assert isinstance(pulse, Readout)
        return self.config.register_acquisition_pulse(channel, pulse)

    def register_pulses(self, configs: dict[str, Config], sequence: PulseSequence):
        """Adds all pulses except measurements of a given sequence in the QM
        ``config``.

        Returns:
            acquisitions (dict): Map from measurement instructions to acquisition objects.
        """
        for id, pulse in sequence:
            if hasattr(pulse, "duration") and not pulse.duration.is_integer():
                raise ValueError(
                    f"Quantum Machines cannot play pulse with duration {pulse.duration}. "
                    "Only integer duration in ns is supported."
                )

            if isinstance(pulse, Pulse):
                self.register_pulse(id, pulse)
            elif isinstance(pulse, Readout):
                self.register_pulse(id, pulse)

    def register_duration_sweeper_pulses(
        self, args: ExecutionArguments, sweeper: Sweeper
    ):
        """Register pulse with many different durations.

        Needed when sweeping duration.
        """
        for pulse in sweeper.pulses:
            if isinstance(pulse, (Align, Delay)):
                continue

            params = args.parameters[operation(pulse)]
            channel_ids = args.sequence.pulse_channels(pulse.id)
            channel = self.channels[channel_ids[0]].logical_channel
            original_pulse = (
                pulse if params.amplitude_pulse is None else params.amplitude_pulse
            )
            for value in sweeper.values:
                sweep_pulse = original_pulse.model_copy(update={"duration": value})
                sweep_op = self.register_pulse(channel, sweep_pulse)
                params.duration_ops.append((value, sweep_op))

    def register_amplitude_sweeper_pulses(
        self, args: ExecutionArguments, sweeper: Sweeper
    ):
        """Register pulse with different amplitude.

        Needed when sweeping amplitude because the original amplitude
        may not sufficient to reach all the sweeper values.
        """
        amplitude = sweeper_amplitude(sweeper.values)
        for pulse in sweeper.pulses:
            channel_ids = args.sequence.pulse_channels(pulse.id)
            channel = self.channels[channel_ids[0]].logical_channel
            sweep_pulse = pulse.model_copy(update={"amplitude": amplitude})

            params = args.parameters[operation(pulse)]
            params.amplitude_pulse = sweep_pulse
            params.amplitude_op = self.register_pulse(channel, sweep_pulse)

    def register_acquisitions(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Add all measurements of a given sequence in the QM ``config``.

        Returns:
            acquisitions (dict): Map from measurement instructions to acquisition objects.
        """
        acquisitions = {}
        for channel_id, readout in sequence:
            if not isinstance(readout, Readout):
                continue

            if readout.probe.duration != readout.acquisition.duration:
                raise ValueError(
                    "Quantum Machines does not support acquisition with different duration than probe."
                )

            op = self.config.register_acquisition_pulse(channel_id, readout)

            acq_config = configs[channel_id]
            assert isinstance(acq_config, QmAcquisitionConfig)
            self.config.register_integration_weights(
                channel_id, readout.duration, acq_config.kernel
            )
            if (op, channel_id) in acquisitions:
                acquisition = acquisitions[(op, channel_id)]
            else:
                acquisition = acquisitions[(op, channel_id)] = create_acquisition(
                    op, channel_id, options, acq_config.threshold, acq_config.iq_angle
                )
            acquisition.keys.append(readout.acquisition.id)

        return acquisitions

    def preprocess_sweeps(
        self,
        sweepers: list[ParallelSweepers],
        configs: dict[str, Config],
        args: ExecutionArguments,
    ):
        """Preprocessing and checks needed before executing some sweeps.

        Amplitude and duration sweeps require registering additional pulses in the QM ``config.
        """
        for sweeper in find_sweepers(sweepers, Parameter.frequency):
            check_frequency_bandwidth(sweeper.channels, configs, sweeper.values)
        for sweeper in find_sweepers(sweepers, Parameter.amplitude):
            self.register_amplitude_sweeper_pulses(args, sweeper)
        for sweeper in find_sweepers(sweepers, Parameter.duration):
            self.register_duration_sweeper_pulses(args, sweeper)

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language."""
        machine = self.manager.open_qm(asdict(self.config))
        return machine.execute(program)

    def simulate_program(self, program):
        """Simulates an arbitrary program written in QUA language."""
        ncontrollers = len(self.config.controllers)
        controller_connections = create_simulator_controller_connections(ncontrollers)
        simulation_config = SimulationConfig(
            duration=self.simulation_duration,
            controller_connections=controller_connections,
        )
        return self.manager.simulate(asdict(self.config), program, simulation_config)

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ):
        if len(sequences) == 0:
            return {}
        elif len(sequences) == 1:
            sequence = sequences[0]
        else:
            sequence, _ = unroll_sequences(sequences, options.relaxation_time)

        if len(sequence) == 0:
            return {}

        # register DC elements so that all qubits are
        # sweetspot even when they are not used
        for id, channel in self.channels.items():
            if isinstance(channel, DcChannel):
                self.configure_channel(id, configs)

        self.configure_channels(configs, sequence.channels)
        self.register_pulses(configs, sequence)
        acquisitions = self.register_acquisitions(configs, sequence, options)

        args = ExecutionArguments(sequence, acquisitions, options.relaxation_time)
        self.preprocess_sweeps(sweepers, configs, args)
        experiment = program(configs, args, options, sweepers)

        if self.script_file_name is not None:
            script_config = (
                {"version": 1} if self.manager is None else asdict(self.config)
            )
            script = generate_qua_script(experiment, script_config)
            with open(self.script_file_name, "w") as file:
                file.write(script)

        if self.manager is None:
            warnings.warn(
                "Not connected to Quantum Machines. Returning program and config."
            )
            return {"program": experiment, "config": asdict(self.config)}

        if self.simulation_duration is not None:
            result = self.simulate_program(experiment)
            results = {}
            for _, pulse in sequence:
                if isinstance(pulse, Acquisition):
                    results[pulse.id] = result
            return results

        result = self.execute_program(experiment)
        return fetch_results(result, acquisitions.values())
