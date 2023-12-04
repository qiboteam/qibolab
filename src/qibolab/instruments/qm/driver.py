from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional

from qm import generate_qua_script, qua
from qm.qua import declare, for_
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab import AveragingMode
from qibolab.instruments.abstract import Controller

from .config import IQPortId, QMConfig, QMPort
from .sequence import Sequence
from .sweepers import sweep


@dataclass
class QMOPX(Controller):
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

    PortType: ClassVar = QMPort

    name: str
    address: str

    manager: Optional[QuantumMachinesManager] = None
    """Manager object used for controlling the QM OPXs."""
    config: QMConfig = field(default_factory=QMConfig)
    """Configuration dictionary required for pulse execution on the OPXs."""
    is_connected: bool = False
    """Boolean that shows whether we are connected to the QM manager."""
    time_of_flight: int = 0
    """Time of flight used for hardware signal integration."""
    smearing: int = 0
    """Smearing used for hardware signal integration."""
    _ports: Dict[IQPortId, QMPort] = field(default_factory=dict)
    """Dictionary holding the ports of controllers that are connected."""
    script_file_name: Optional[str] = "qua_script.txt"
    """Name of the file that the QUA program will dumped in that after every execution.
    If ``None`` the program will not be dumped.
    """

    def __post_init__(self):
        super().__init__(self.name, self.address)

    def connect(self):
        """Connect to the QM manager."""
        host, port = self.address.split(":")
        self.manager = QuantumMachinesManager(host, int(port))

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
            results[pulse.qubit] = results[pulse.serial] = qmpulse.acquisition.fetch(handles)
        return results

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
                self.config.register_flux_element(qubit)

        qmsequence = Sequence.create(qubits, sequence, sweepers, self.config, self.time_of_flight, self.smearing)
        # play pulses using QUA
        with qua.program() as experiment:
            n = declare(int)
            for qmpulse in qmsequence.ro_pulses:
                threshold = qubits[qmpulse.pulse.qubit].threshold
                iq_angle = qubits[qmpulse.pulse.qubit].iq_angle
                qmpulse.declare_output(options, threshold, iq_angle)

            with for_(n, 0, n < options.nshots, n + 1):
                sweep(list(sweepers), qubits, qmsequence, options.relaxation_time, self.config)

            with qua.stream_processing():
                for qmpulse in qmsequence.ro_pulses:
                    qmpulse.acquisition.download(*buffer_dims)

        if self.script_file_name is not None:
            with open(self.script_file_name, "w") as file:
                file.write(generate_qua_script(experiment, self.config.__dict__))

        result = self.execute_program(experiment)
        return self.fetch_results(result, qmsequence.ro_pulses)
