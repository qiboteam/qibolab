import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.octave import QmOctaveConfig
from qm.simulate.credentials import create_credentials
from qualang_tools.simulator_tools import create_simulator_controller_connections

from qibolab.components import Channel, Config, DcChannel, IqChannel
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.pulses import Delay, Pulse, PulseSequence, VirtualZ
from qibolab.sweeper import ParallelSweepers, Parameter
from qibolab.unrolling import Bounds

from .components import QmChannel
from .config import SAMPLING_RATE, QmConfig, operation
from .program import create_acquisition, program

OCTAVE_ADDRESS_OFFSET = 11000
"""Offset to be added to Octave addresses, because they must be 11xxx, where
xxx are the last three digits of the Octave IP address."""

__all__ = ["QmController", "Octave"]


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


def find_baking_pulses(sweepers):
    """Find pulses that require baking because we are sweeping their duration.

    Args:
        sweepers (list): List of :class:`qibolab.sweeper.Sweeper` objects.
    """
    to_bake = set()
    for sweeper in sweepers:
        values = sweeper.values
        step = values[1] - values[0] if len(values) > 0 else values[0]
        if sweeper.parameter is Parameter.duration and step % 4 != 0:
            for pulse in sweeper.pulses:
                to_bake.add(pulse.id)

    return to_bake


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


@dataclass
class QmController(Controller):
    """:class:`qibolab.instruments.abstract.Controller` object for controlling
    a Quantum Machines cluster.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language.
    The ``config`` file is generated using the ``dataclass`` objects defined in
    :py_mod:`qibolab.instruments.qm.config`.
    The QUA program is generated using the methods in :py_mod:`qibolab.instruments.qm.program`.
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

    bounds: Bounds = Bounds(0, 0, 0)
    """Maximum bounds used for batching in sequence unrolling."""
    calibration_path: Optional[str] = None
    """Path to the JSON file that contains the mixer calibration."""
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
        # redefine bounds because abstract instrument overwrites them
        # FIXME: This overwrites the ``bounds`` given in the runcard!
        self.bounds = Bounds(
            waveforms=int(4e4),
            readout=30,
            instructions=int(1e6),
        )

        if self.simulation_duration is not None:
            # convert simulation duration from ns to clock cycles
            self.simulation_duration //= 4

    @property
    def sampling_rate(self):
        """Sampling rate of Quantum Machines instruments."""
        return SAMPLING_RATE

    def connect(self):
        """Connect to the Quantum Machines manager."""
        host, port = self.address.split(":")
        octave = declare_octaves(self.octaves, host, self.calibration_path)
        credentials = None
        if self.cloud:
            credentials = create_credentials()
        self.manager = QuantumMachinesManager(
            host=host, port=int(port), octave=octave, credentials=credentials
        )
        self.is_connected = True

    def disconnect(self):
        """Disconnect from QM manager."""
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
        channel_config = configs[logical_channel.name]
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

        else:
            raise TypeError(f"Unknown channel type: {type(channel)}.")

    def register_pulse(self, channel: Channel, pulse: Pulse):
        """Add pulse in the ``config``."""
        # if (
        #    pulse.duration % 4 != 0
        #    or pulse.duration < 16
        #    or pulse.id in pulses_to_bake
        # ):
        #    qmpulse = BakedPulse(pulse, element)
        #    qmpulse.bake(self.config, durations=[pulse.duration])
        # else:
        if isinstance(channel, DcChannel):
            return self.config.register_dc_pulse(channel.name, pulse)
        if channel.acquisition is None:
            return self.config.register_iq_pulse(channel.name, pulse)
        return self.config.register_acquisition_pulse(channel.name, pulse)

    def register_pulses(self, configs, sequence, integration_setup, options):
        """Adds all pulses of a given :class:`qibolab.pulses.PulseSequence` in
        the ``config``.

        Returns:
            acquisitions (dict): Map from measurement instructions to acquisition objects.
        """
        acquisitions = {}
        for name, channel_sequence in sequence.items():
            channel = self.channels[name]
            self.configure_channel(channel, configs)

            for pulse in channel_sequence:
                if isinstance(pulse, (Delay, VirtualZ)):
                    continue

                logical_channel = channel.logical_channel
                op = self.register_pulse(logical_channel, pulse)

                if (
                    isinstance(logical_channel, IqChannel)
                    and logical_channel.acquisition is not None
                ):
                    kernel, threshold, iq_angle = integration_setup[
                        logical_channel.acquisition
                    ]
                    self.config.register_integration_weights(
                        name, pulse.duration, kernel
                    )
                    if (op, name) in acquisitions:
                        acquisition = acquisitions[(op, name)]
                    else:
                        acquisition = acquisitions[(op, name)] = create_acquisition(
                            op,
                            name,
                            options,
                            threshold,
                            iq_angle,
                        )
                    acquisition.keys.append(pulse.id)

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
        integration_setup,
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
        acquisitions = self.register_pulses(
            configs, sequence, integration_setup, options
        )
        experiment = program(configs, sequence, options, acquisitions, sweepers)

        if self.manager is None:
            warnings.warn(
                "Not connected to Quantum Machines. Returning program and config."
            )
            return {"program": experiment, "config": asdict(self.config)}

        if self.script_file_name is not None:
            script = generate_qua_script(experiment, asdict(self.config))
            for pulses in sequence.values():
                for pulse in pulses:
                    script = script.replace(operation(pulse), str(pulse))
            with open(self.script_file_name, "w") as file:
                file.write(script)

        if self.simulation_duration is not None:
            result = self.simulate_program(experiment)
            results = {}
            for channel_name, pulses in sequence.items():
                if self.channels[channel_name].logical_channel.acquisition is not None:
                    for pulse in pulses:
                        results[pulse.id] = result
            return results

        result = self.execute_program(experiment)
        return fetch_results(result, acquisitions.values())
