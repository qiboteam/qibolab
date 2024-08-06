import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script, qua
from qm.octave import QmOctaveConfig
from qm.qua import declare, for_
from qm.simulate.credentials import create_credentials
from qualang_tools.simulator_tools import create_simulator_controller_connections

from qibolab import AveragingMode
from qibolab.instruments.abstract import Controller
from qibolab.pulses import PulseType
from qibolab.sweeper import Parameter
from qibolab.unrolling import Bounds

from .acquisition import declare_acquisitions, fetch_results
from .config import SAMPLING_RATE, QMConfig
from .devices import Octave, OPXplus
from .ports import OPXIQ
from .sequence import BakedPulse, QMPulse, Sequence
from .sweepers import sweep

OCTAVE_ADDRESS_OFFSET = 11000
"""Offset to be added to Octave addresses, because they must be 11xxx, where
xxx are the last three digits of the Octave IP address."""
CALIBRATION_DB = "calibration_db.json"
"""Name of the file where the mixer calibration is stored."""


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


def controllers_config(qubits, time_of_flight, smearing=0):
    """Create a Quantum Machines configuration without pulses.

    This contains the readout and drive elements and controllers and
    is used by :meth:`qibolab.instruments.qm.controller.QMController.calibrate_mixers`.

    Args:
        qubits (list): List of :class:`qibolab.qubits.Qubit` objects to be
            included in the config.
        time_of_flight (int): Time of flight used on readout elements.
        smearing (int): Smearing used on readout elements.
    """
    config = QMConfig()
    for qubit in qubits:
        if qubit.readout is not None:
            config.register_port(qubit.readout.port)
            config.register_readout_element(
                qubit, qubit.mixer_frequencies["MZ"][1], time_of_flight, smearing
            )
        if qubit.drive is not None:
            config.register_port(qubit.drive.port)
            config.register_drive_element(qubit, qubit.mixer_frequencies["RX"][1])
    return config


@dataclass
class QMController(Controller):
    """:class:`qibolab.instruments.abstract.Controller` object for controlling
    a Quantum Machines cluster.

    A cluster consists of multiple :class:`qibolab.instruments.qm.devices.QMDevice` devices.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language.
    The ``config`` file is generated in parts in :class:`qibolab.instruments.qm.config.QMConfig`.
    Controllers, elements and pulses are all registered after a pulse sequence is given, so that
    the config contains only elements related to the participating qubits.
    The QUA program for executing an arbitrary :class:`qibolab.pulses.PulseSequence` is written in
    :meth:`qibolab.instruments.qm.controller.QMController.play` and executed in
    :meth:`qibolab.instruments.qm.controller.QMController.execute_program`.
    """

    name: str
    """Name of the instrument instance."""
    address: str
    """IP address and port for connecting to the OPX instruments.

    Has the form XXX.XXX.XXX.XXX:XXX.
    """

    opxs: dict[int, OPXplus] = field(default_factory=dict)
    """Dictionary containing the
    :class:`qibolab.instruments.qm.devices.OPXplus` instruments being used."""
    octaves: dict[int, Octave] = field(default_factory=dict)
    """Dictionary containing the :class:`qibolab.instruments.qm.devices.Octave`
    instruments being used."""

    time_of_flight: int = 0
    """Time of flight used for hardware signal integration."""
    smearing: int = 0
    """Smearing used for hardware signal integration."""
    bounds: Bounds = Bounds(0, 0, 0)
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
    config: QMConfig = field(default_factory=QMConfig)
    """Configuration dictionary required for pulse execution on the OPXs."""
    is_connected: bool = False
    """Boolean that shows whether we are connected to the QM manager."""

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
        self.bounds = Bounds(
            waveforms=int(4e4),
            readout=30,
            instructions=int(1e6),
        )
        # convert lists to dicts
        if not isinstance(self.opxs, dict):
            self.opxs = {instr.name: instr for instr in self.opxs}
        if not isinstance(self.octaves, dict):
            self.octaves = {instr.name: instr for instr in self.octaves}

        if self.simulation_duration is not None:
            # convert simulation duration from ns to clock cycles
            self.simulation_duration //= 4

    def ports(self, name, output=True):
        """Provides instrument ports to the user.

        Note that individual ports can also be accessed from the corresponding devices
        using :meth:`qibolab.instruments.qm.devices.QMDevice.ports`.

        Args:
            name (tuple): Contains the numbers of controller and port to be obtained.
                For example ``((conX, Y),)`` returns port-Y of OPX+ controller X.
                ``((conX, Y), (conX, Z))`` returns port-Y and Z of OPX+ controller X
                as an :class:`qibolab.instruments.qm.ports.OPXIQ` port pair.
            output (bool): ``True`` for obtaining an output port, otherwise an
                input port is returned. Default is ``True``.
        """
        if len(name) == 1:
            con, port = name[0]
            return self.opxs[con].ports(port, output)
        elif len(name) == 2:
            (con1, port1), (con2, port2) = name
            return OPXIQ(
                self.opxs[con1].ports(port1, output),
                self.opxs[con2].ports(port2, output),
            )
        else:
            raise ValueError(f"Invalid port {name} for Quantum Machines controller.")

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

    def calibrate_mixers(self, qubits):
        """Calibrate Octave mixers for readout and drive lines of given qubits.

        Args:
            qubits (list): List of :class:`qibolab.qubits.Qubit` objects for
                which mixers will be calibrated.
        """
        if isinstance(qubits, dict):
            qubits = list(qubits.values())

        config = controllers_config(qubits, self.time_of_flight, self.smearing)
        machine = self.manager.open_qm(config.__dict__)
        for qubit in qubits:
            print(f"Calibrating mixers for qubit {qubit.name}")
            if qubit.readout is not None:
                _lo, _if = qubit.mixer_frequencies["MZ"]
                machine.calibrate_element(f"readout{qubit.name}", {_lo: (_if,)})
            if qubit.drive is not None:
                _lo, _if = qubit.mixer_frequencies["RX"]
                machine.calibrate_element(f"drive{qubit.name}", {_lo: (_if,)})

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.
        """
        machine = self.manager.open_qm(self.config.__dict__)
        return machine.execute(program)

    def simulate_program(self, program):
        """Simulates an arbitrary program written in QUA language.

        Args:
            program: QUA program.
        """
        ncontrollers = len(self.config.controllers)
        controller_connections = create_simulator_controller_connections(ncontrollers)
        simulation_config = SimulationConfig(
            duration=self.simulation_duration,
            controller_connections=controller_connections,
        )
        return self.manager.simulate(self.config.__dict__, program, simulation_config)

    def create_sequence(self, qubits, sequence, sweepers):
        """Translates a :class:`qibolab.pulses.PulseSequence` to a
        :class:`qibolab.instruments.qm.sequence.Sequence`.

        Args:
            qubits (list): List of :class:`qibolab.platforms.abstract.Qubit` objects
                passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to translate.
            sweepers (list): List of sweeper objects so that pulses that require baking are identified.
        Returns:
            (:class:`qibolab.instruments.qm.sequence.Sequence`) containing the pulses from given pulse sequence.
        """
        # Current driver cannot play overlapping pulses on drive and flux channels
        # If we want to play overlapping pulses we need to define different elements on the same ports
        # like we do for readout multiplex

        pulses_to_bake = find_baking_pulses(sweepers)

        qmsequence = Sequence()
        ro_pulses = []
        for pulse in sorted(sequence, key=lambda pulse: (pulse.start, pulse.duration)):
            qubit = qubits[pulse.qubit]

            self.config.register_port(getattr(qubit, pulse.type.name.lower()).port)
            if pulse.type is PulseType.READOUT:
                self.config.register_port(qubit.feedback.port)

            element = self.config.register_element(
                qubit, pulse, self.time_of_flight, self.smearing
            )
            if (
                pulse.duration % 4 != 0
                or pulse.duration < 16
                or pulse.id in pulses_to_bake
            ):
                qmpulse = BakedPulse(pulse, element)
                qmpulse.bake(self.config, durations=[pulse.duration])
            else:
                qmpulse = QMPulse(pulse, element)
                if pulse.type is PulseType.READOUT:
                    ro_pulses.append(qmpulse)
                self.config.register_pulse(qubit, qmpulse)
            qmsequence.add(qmpulse)

        qmsequence.shift()
        return qmsequence, ro_pulses

    def play(self, qubits, couplers, sequence, options):
        return self.sweep(qubits, couplers, sequence, options)

    def sweep(self, qubits, couplers, sequence, options, *sweepers):
        if not sequence:
            return {}

        buffer_dims = [len(sweeper.values) for sweeper in reversed(sweepers)]
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            buffer_dims.append(options.nshots)

        # register flux elements for all qubits so that they are
        # always at sweetspot even when they are not used
        for qubit in qubits.values():
            if qubit.flux:
                self.config.register_port(qubit.flux.port)
                self.config.register_flux_element(qubit)

        qmsequence, ro_pulses = self.create_sequence(qubits, sequence, sweepers)
        # play pulses using QUA
        with qua.program() as experiment:
            n = declare(int)
            acquisitions = declare_acquisitions(ro_pulses, qubits, options)
            with for_(n, 0, n < options.nshots, n + 1):
                sweep(
                    list(sweepers),
                    qubits,
                    qmsequence,
                    options.relaxation_time,
                    self.config,
                )

            with qua.stream_processing():
                for acquisition in acquisitions:
                    acquisition.download(*buffer_dims)

        if self.script_file_name is not None:
            with open(self.script_file_name, "w") as file:
                file.write(generate_qua_script(experiment, self.config.__dict__))

        if self.simulation_duration is not None:
            result = self.simulate_program(experiment)
            results = {}
            for qmpulse in ro_pulses:
                pulse = qmpulse.pulse
                results[pulse.qubit] = results[pulse.id] = result
            return results
        else:
            result = self.execute_program(experiment)
            return fetch_results(result, acquisitions)
