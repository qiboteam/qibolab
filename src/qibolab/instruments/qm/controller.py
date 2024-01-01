from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from qm import generate_qua_script, qua
from qm.octave import QmOctaveConfig
from qm.qua import declare, for_
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab import AveragingMode
from qibolab.instruments.abstract import Controller
from qibolab.pulses import PulseType
from qibolab.sweeper import Parameter

from .config import QMConfig
from .devices import Octave, OPXplus
from .ports import OPXIQ, OctaveInput, OctaveOutput, OPXInput, OPXOutput
from .sequence import BakedPulse, QMPulse, Sequence
from .sweepers import sweep

IQPortId = Tuple[Tuple[str, int], Tuple[str, int]]

OCTAVE_ADDRESS = 11000
"""Must be 11xxx, where xxx are the last three digits of the Octave IP
address."""


def declare_octaves(octaves, host):
    """Initiate octave_config class, set the calibration file and add octaves
    info.

    :param octaves: objects that holds the information about octave's
        name, the controller that is connected to this octave, octave's
        ip and octave's port.
    """
    # TODO: Fix docstring
    config = None
    if len(octaves) > 0:
        config = QmOctaveConfig()
        # config.set_calibration_db(os.getcwd())
        for octave in octaves:
            config.add_device_info(octave.name, host, OCTAVE_ADDRESS + octave.port)
    return config


def find_duration_sweeper_pulses(sweepers):
    """Find all pulses that require baking because we are sweeping their
    duration."""
    duration_sweep_pulses = set()
    for sweeper in sweepers:
        try:
            step = sweeper.values[1] - sweeper.values[0]
        except IndexError:
            step = sweeper.values[0]

        if sweeper.parameter is Parameter.duration and step % 4 != 0:
            for pulse in sweeper.pulses:
                duration_sweep_pulses.add(pulse.serial)

    return duration_sweep_pulses


@dataclass
class QMController(Controller):
    """Instrument object for controlling Quantum Machines (QM) OPX controllers.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language. The ``config`` file is generated in parts in the following places
    in the ``register_*`` methods. The controllers, elements and pulses are all
    registered after a pulse sequence is given, so that the config contains only
    elements related to the participating qubits.
    The QUA program for executing an arbitrary qibolab ``PulseSequence`` is written in
    ``play`` and ``play_pulses`` and executed in ``execute_program``.

    Args:
        name (str): Name of the instrument instance.
        address (str): IP address and port for connecting to the OPX instruments.
    """

    name: str
    address: str

    opxs: Dict[int, OPXplus] = field(default_factory=dict)
    octaves: Dict[int, Octave] = field(default_factory=dict)

    time_of_flight: int = 0
    """Time of flight used for hardware signal integration."""
    smearing: int = 0
    """Smearing used for hardware signal integration."""
    script_file_name: Optional[str] = "qua_script.py"
    """Name of the file that the QUA program will dumped in that after every
    execution.

    If ``None`` the program will not be dumped.
    """

    output_ports: Dict[IQPortId, OPXIQ] = field(default_factory=dict)
    input_ports: Dict[IQPortId, OPXIQ] = field(default_factory=dict)
    """Dictionary holding the ports of IQ pairs.

    Not needed when using only Octaves.
    """

    manager: Optional[QuantumMachinesManager] = None
    """Manager object used for controlling the QM OPXs."""
    config: QMConfig = field(default_factory=QMConfig)
    """Configuration dictionary required for pulse execution on the OPXs."""
    is_connected: bool = False
    """Boolean that shows whether we are connected to the QM manager."""

    def __post_init__(self):
        super().__init__(self.name, self.address)
        if isinstance(self.opxs, list):
            self.opxs = {instr.name: instr for instr in self.opxs}
        if isinstance(self.octaves, list):
            self.octaves = {instr.name: instr for instr in self.octaves}

    def ports(self, name, input=False):
        if len(name) != 2:
            raise ValueError(
                "QMController provides only IQ ports. Please access individual ports from the specific device."
            )
        _ports = self.input_ports if input else self.output_ports
        if name not in _ports:
            port_cls = OPXInput if input else OPXOutput
            _ports[name] = OPXIQ(port_cls(*name[0]), port_cls(*name[1]))
        return _ports[name]

    def connect(self):
        """Connect to the QM manager."""
        host, port = self.address.split(":")
        octave = declare_octaves(self.octaves, host)
        self.manager = QuantumMachinesManager(host=host, port=int(port), octave=octave)

    def setup(self):
        """Deprecated method."""
        # controllers are defined when registering pulses
        pass

    def start(self):
        # TODO: Start the OPX flux offsets?
        pass

    def stop(self):
        """Close all running Quantum Machines."""
        # TODO: Use logging
        # log.warn("Closing all Quantum Machines.")
        print("Closing all Quantum Machines.")
        self.manager.close_all_quantum_machines()

    def disconnect(self):
        """Disconnect from QM manager."""
        if self.is_connected:
            self.manager.close()
            self.is_connected = False

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.

        Returns:
            TODO
        """
        machine = self.manager.open_qm(self.config.__dict__)
        return machine.execute(program)

    @staticmethod
    def fetch_results(result, ro_pulses):
        """Fetches results from an executed experiment.

        Defined as ``@staticmethod`` because it is overwritten
        in :class:`qibolab.instruments.qm.simulator.QMSim`.
        """
        # TODO: Update result asynchronously instead of waiting
        # for all values, in order to allow live plotting
        # using ``handles.is_processing()``
        handles = result.result_handles
        handles.wait_for_all_values()
        results = {}
        for qmpulse in ro_pulses:
            pulse = qmpulse.pulse
            results[pulse.qubit] = results[pulse.serial] = qmpulse.acquisition.fetch(
                handles
            )
        return results

    def register_port(self, port):
        if isinstance(port, OPXIQ):
            self.register_port(port.i)
            self.register_port(port.q)
        else:
            self.config.register_port(port)
            if isinstance(port, (OctaveInput, OctaveOutput)):
                self.config.octaves[port.device]["connectivity"] = self.octaves[
                    port.device
                ].connectivity

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

        duration_sweep_pulses = find_duration_sweeper_pulses(sweepers)

        qmsequence = Sequence()
        sort_key = lambda pulse: (pulse.start, pulse.duration)
        for pulse in sorted(sequence.pulses, key=sort_key):
            qubit = qubits[pulse.qubit]

            self.register_port(getattr(qubit, pulse.type.name.lower()).port)
            if pulse.type is PulseType.READOUT:
                self.register_port(qubit.feedback.port)

            self.config.register_element(
                qubit, pulse, self.time_of_flight, self.smearing
            )
            if (
                pulse.duration % 4 != 0
                or pulse.duration < 16
                or pulse.serial in duration_sweep_pulses
            ):
                qmpulse = BakedPulse(pulse)
                qmpulse.bake(self.config, durations=[pulse.duration])
            else:
                qmpulse = QMPulse(pulse)
                self.config.register_pulse(qubit, pulse)
            qmsequence.add(qmpulse)

        qmsequence.shift()
        return qmsequence

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
                self.register_port(qubit.flux.port)
                self.config.register_flux_element(qubit)

        qmsequence = self.create_sequence(qubits, sequence, sweepers)
        # play pulses using QUA
        with qua.program() as experiment:
            n = declare(int)
            for qmpulse in qmsequence.ro_pulses:
                threshold = qubits[qmpulse.pulse.qubit].threshold
                iq_angle = qubits[qmpulse.pulse.qubit].iq_angle
                qmpulse.declare_output(options, threshold, iq_angle)

            with for_(n, 0, n < options.nshots, n + 1):
                sweep(
                    list(sweepers),
                    qubits,
                    qmsequence,
                    options.relaxation_time,
                    self.config,
                )

            with qua.stream_processing():
                for qmpulse in qmsequence.ro_pulses:
                    qmpulse.acquisition.download(*buffer_dims)

        if self.script_file_name is not None:
            with open(self.script_file_name, "w") as file:
                file.write(generate_qua_script(experiment, self.config.__dict__))

        result = self.execute_program(experiment)
        return self.fetch_results(result, qmsequence.ro_pulses)
