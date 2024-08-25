import shutil
import tempfile
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.octave import QmOctaveConfig
from qm.simulate.credentials import create_credentials
from qualang_tools.simulator_tools import create_simulator_controller_connections

from qibolab.components import AcquireChannel, Channel, Config, DcChannel, IqChannel
from qibolab.execution_parameters import ExecutionParameters
from qibolab.identifier import ChannelId
from qibolab.instruments.abstract import Controller
from qibolab.pulses.pulse import Acquisition, Align, Delay, Pulse, _Readout
from qibolab.sequence import PulseSequence
from qibolab.sweeper import ParallelSweepers, Parameter, Sweeper
from qibolab.unrolling import Bounds

from .components import QmChannel
from .config import SAMPLING_RATE, QmConfig, operation
from .program import ExecutionArguments, create_acquisition, program

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


def find_duration_sweepers(sweepers: list[ParallelSweepers]) -> list[Sweeper]:
    """Find duration sweepers in order to register multiple pulses."""
    return [s for ps in sweepers for s in ps if s.parameter is Parameter.duration]


@dataclass
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
    channels: dict[str, QmChannel]

    bounds: str = "qm/bounds"
    """Maximum bounds used for batching in sequence unrolling."""
    calibration_path: Optional[str] = None
    """Path to the JSON file that contains the mixer calibration."""
    write_calibration: bool = False
    """Require writing permissions on calibration DB."""
    _calibration_path: Optional[str] = None
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

    config: QmConfig = field(default_factory=QmConfig)
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

    def __post_init__(self):
        super().__init__(self.name, self.address)
        # convert ``channels`` from list to dict
        self.channels = {
            str(channel.logical_channel.name): channel for channel in self.channels
        }

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

    def configure_channel(self, channel: QmChannel, configs: dict[str, Config]):
        """Add element (QM version of channel) in the config."""
        logical_channel = channel.logical_channel
        channel_config = configs[str(logical_channel.name)]
        self.configure_device(channel.device)

        if isinstance(logical_channel, DcChannel):
            self.config.configure_dc_line(channel, channel_config)

        elif isinstance(logical_channel, IqChannel):
            lo_config = configs[logical_channel.lo]
            if logical_channel.acquisition is None:
                self.config.configure_iq_line(channel, channel_config, lo_config)

            else:
                acquisition = logical_channel.acquisition
                acquire_channel = self.channels[acquisition]
                self.configure_device(acquire_channel.device)
                self.config.configure_acquire_line(
                    acquire_channel,
                    channel,
                    configs[acquisition],
                    channel_config,
                    lo_config,
                )

        elif not isinstance(logical_channel, AcquireChannel):
            raise TypeError(f"Unknown channel type: {type(logical_channel)}.")

    def configure_channels(
        self,
        configs: dict[str, Config],
        channels: set[ChannelId],
    ):
        """Register channels participating in the sequence in the QM
        ``config``."""
        for channel_id in channels:
            channel = self.channels[str(channel_id)]
            self.configure_channel(channel, configs)

    def register_pulse(self, channel: Channel, pulse: Pulse) -> str:
        """Add pulse in the QM ``config`` and return corresponding
        operation."""
        name = str(channel.name)
        if isinstance(channel, DcChannel):
            return self.config.register_dc_pulse(name, pulse)
        if channel.acquisition is None:
            return self.config.register_iq_pulse(name, pulse)
        return self.config.register_acquisition_pulse(name, pulse)

    def register_pulses(self, configs: dict[str, Config], sequence: PulseSequence):
        """Adds all pulses except measurements of a given sequence in the QM
        ``config``.

        Returns:
            acquisitions (dict): Map from measurement instructions to acquisition objects.
        """
        for channel_id, pulse in sequence:
            if (
                hasattr(pulse, "duration")
                and isinstance(pulse.duration, float)
                and not pulse.duration.is_integer()
            ):
                raise ValueError(
                    f"Quantum Machines cannot play pulse with duration {pulse.duration}. "
                    "Only integer duration in ns is supported."
                )
            if isinstance(pulse, Pulse):
                channel = self.channels[str(channel_id)].logical_channel
                self.register_pulse(channel, pulse)

    def register_duration_sweeper_pulses(
        self, args: ExecutionArguments, sweeper: Sweeper
    ):
        """Register pulse with many different durations, in order to sweep
        duration."""
        for pulse in sweeper.pulses:
            if isinstance(pulse, (Align, Delay)):
                continue

            op = operation(pulse)
            channel_name = str(args.sequence.pulse_channels(pulse.id)[0])
            channel = self.channels[channel_name].logical_channel
            for value in sweeper.values:
                sweep_pulse = pulse.model_copy(update={"duration": value})
                sweep_op = self.register_pulse(channel, sweep_pulse)
                args.parameters[op].pulses.append((value, sweep_op))

    def register_acquisitions(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Adds all measurements of a given sequence in the QM ``config``.

        Returns:
            acquisitions (dict): Map from measurement instructions to acquisition objects.
        """
        acquisitions = {}
        for channel_id, readout in sequence.as_readouts:
            if not isinstance(readout, _Readout):
                continue

            if readout.probe.duration != readout.acquisition.duration:
                raise ValueError(
                    "Quantum Machines does not support acquisition with different duration than probe."
                )

            channel_name = str(channel_id)
            channel = self.channels[channel_name].logical_channel
            op = self.config.register_acquisition_pulse(channel_name, readout.probe)

            acq_config = configs[channel.acquisition]
            self.config.register_integration_weights(
                channel_name, readout.duration, acq_config.kernel
            )
            if (op, channel_name) in acquisitions:
                acquisition = acquisitions[(op, channel_name)]
            else:
                acquisition = acquisitions[(op, channel_name)] = create_acquisition(
                    op,
                    channel_name,
                    options,
                    acq_config.threshold,
                    acq_config.iq_angle,
                )
            acquisition.keys.append(readout.acquisition.id)

        return acquisitions

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
        if len(sequences) > 1:
            raise NotImplementedError
        elif len(sequences) == 0 or len(sequences[0]) == 0:
            return {}

        # register DC elements so that all qubits are
        # sweetspot even when they are not used
        for channel in self.channels.values():
            if isinstance(channel.logical_channel, DcChannel):
                self.configure_channel(channel, configs)

        sequence = sequences[0]
        self.configure_channels(configs, sequence.channels)
        self.register_pulses(configs, sequence)
        acquisitions = self.register_acquisitions(configs, sequence, options)

        args = ExecutionArguments(sequence, acquisitions, options.relaxation_time)

        for sweeper in find_duration_sweepers(sweepers):
            self.register_duration_sweeper_pulses(args, sweeper)

        experiment = program(configs, args, options, sweepers)

        if self.manager is None:
            warnings.warn(
                "Not connected to Quantum Machines. Returning program and config."
            )
            return {"program": experiment, "config": asdict(self.config)}

        if self.script_file_name is not None:
            script = generate_qua_script(experiment, asdict(self.config))
            with open(self.script_file_name, "w") as file:
                file.write(script)

        if self.simulation_duration is not None:
            result = self.simulate_program(experiment)
            results = {}
            for _, pulse in sequence:
                if isinstance(pulse, Acquisition):
                    results[pulse.id] = result
            return results

        result = self.execute_program(experiment)
        return fetch_results(result, acquisitions.values())
