import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional

import numpy as np
from qibo.config import raise_error
from qm import generate_qua_script, qua
from qm.qua import declare, fixed, for_
from qm.QuantumMachinesManager import QuantumMachinesManager
from qualang_tools.loops import from_array

from qibolab import AveragingMode
from qibolab.channels import check_max_offset
from qibolab.instruments.abstract import Controller
from qibolab.instruments.qm.config import IQPortId, QMConfig, QMPort
from qibolab.instruments.qm.sequence import BakedPulse, QMPulse, Sequence
from qibolab.pulses import PulseType
from qibolab.sweeper import Parameter


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

    def create_qmsequence(self, qubits, sequence):
        """Translates a :class:`qibolab.pulses.PulseSequence` to a :class:`qibolab.instruments.qm.Sequence`.

        Also register flux elements for all qubits (if applicable) so that all qubits are operated at their
        sweetspot.

        Args:
            qubits (list): List of :class:`qibolab.platforms.abstract.Qubit` objects
                passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to translate.
        Returns:
            (:class:`qibolab.instruments.qm.Sequence`) containing the pulses from given pulse sequence.
        """
        # register flux elements for all qubits so that they are
        # always at sweetspot even when they are not used
        for qubit in qubits.values():
            if qubit.flux:
                self.config.register_flux_element(qubit)

        # Current driver cannot play overlapping pulses on drive and flux channels
        # If we want to play overlapping pulses we need to define different elements on the same ports
        # like we do for readout multiplex
        qmsequence = Sequence()
        for pulse in sorted(sequence.pulses, key=lambda pulse: (pulse.start, pulse.duration)):
            if pulse.type is PulseType.FLUX:
                # register flux element (if it does not already exist)
                self.config.register_flux_element(qubits[pulse.qubit], pulse.frequency)
                qmpulse = BakedPulse(pulse)
                qmpulse.bake(self.config, durations=[pulse.duration])
                qmsequence.add(qmpulse)
            else:
                qmpulse = QMPulse(pulse)
                qmsequence.add(qmpulse)
                if pulse.duration % 4 != 0 or pulse.duration < 16:
                    raise_error(NotImplementedError, "1ns resolution is available for flux pulses only.")
                self.config.register_pulse(qubits[pulse.qubit], pulse, self.time_of_flight, self.smearing)

        qmsequence.shift()

        return qmsequence

    @staticmethod
    def play_pulses(qmsequence, relaxation_time=0):
        """Part of QUA program that plays an arbitrary pulse sequence.

        Should be used inside a ``program()`` context.

        Args:
            qmsequence (list): Pulse sequence containing QM specific pulses (``qmpulse``).
                These pulses are defined in :meth:`qibolab.instruments.qm.QMOPX.play` and
                hold attributes for the ``element`` and ``operation`` that corresponds to
                each pulse, as defined in the QM config.
        """
        needs_reset = False
        qua.align()
        for qmpulse in qmsequence.qmpulses:
            pulse = qmpulse.pulse
            if qmpulse.wait_cycles is not None:
                qua.wait(qmpulse.wait_cycles, qmpulse.element)
            if pulse.type is PulseType.READOUT:
                qmpulse.acquisition.measure(qmpulse.operation, qmpulse.element)
            else:
                if not isinstance(qmpulse.relative_phase, float) or qmpulse.relative_phase != 0:
                    qua.frame_rotation_2pi(qmpulse.relative_phase, qmpulse.element)
                    needs_reset = True
                qmpulse.play()
                if needs_reset:
                    qua.reset_frame(qmpulse.element)
                    needs_reset = False
                if len(qmpulse.elements_to_align) > 1:
                    qua.align(*qmpulse.elements_to_align)

        # for Rabi-length?
        if relaxation_time > 0:
            qua.wait(relaxation_time // 4)

        # Save data to the stream processing
        for qmpulse in qmsequence.ro_pulses:
            qmpulse.acquisition.save()

    @staticmethod
    def fetch_results(result, ro_pulses):
        """Fetches results from an executed experiment."""
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

    def play(self, qubits, sequence, options):
        return self.sweep(qubits, sequence, options)

    def sweep(self, qubits, sequence, options, *sweepers):
        if not sequence:
            return {}

        buffer_dims = [len(sweeper.values) for sweeper in reversed(sweepers)]
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            buffer_dims.append(options.nshots)

        qmsequence = self.create_qmsequence(qubits, sequence)
        # play pulses using QUA
        with qua.program() as experiment:
            n = declare(int)
            for qmpulse in qmsequence.ro_pulses:
                threshold = qubits[qmpulse.pulse.qubit].threshold
                iq_angle = qubits[qmpulse.pulse.qubit].iq_angle
                qmpulse.declare_output(options, threshold, iq_angle)

            with for_(n, 0, n < options.nshots, n + 1):
                self.sweep_recursion(list(sweepers), qubits, qmsequence, options.relaxation_time)

            with qua.stream_processing():
                for qmpulse in qmsequence.ro_pulses:
                    qmpulse.acquisition.download(*buffer_dims)

        if self.script_file_name is not None:
            with open(self.script_file_name, "w") as file:
                file.write(generate_qua_script(experiment, self.config.__dict__))

        result = self.execute_program(experiment)
        return self.fetch_results(result, qmsequence.ro_pulses)

    def play_sequences(self, qubits, sequence, options):
        raise NotImplementedError

    @staticmethod
    def maximum_sweep_value(values, value0):
        """Calculates maximum value that is reached during a sweep.

        Useful to check whether a sweep exceeds the range of allowed values.
        Note that both the array of values we sweep and the center value can
        be negative, so we need to make sure that the maximum absolute value
        is within range.

        Args:
            values (np.ndarray): Array of values we will sweep over.
            value0 (float, int): Center value of the sweep.
        """
        return max(abs(min(values) + value0), abs(max(values) + value0))

    def sweep_frequency(self, sweepers, qubits, qmsequence, relaxation_time):
        from qm.qua import update_frequency

        sweeper = sweepers[0]
        freqs0 = []
        for pulse in sweeper.pulses:
            qubit = qubits[pulse.qubit]
            if pulse.type is PulseType.DRIVE:
                lo_frequency = math.floor(qubit.drive.local_oscillator.frequency)
            elif pulse.type is PulseType.READOUT:
                lo_frequency = math.floor(qubit.readout.local_oscillator.frequency)
            else:
                raise_error(NotImplementedError, f"Cannot sweep frequency of pulse of type {pulse.type}.")
            # convert to IF frequency for readout and drive pulses
            f0 = math.floor(pulse.frequency - lo_frequency)
            freqs0.append(declare(int, value=f0))
            # check if sweep is within the supported bandwidth [-400, 400] MHz
            max_freq = self.maximum_sweep_value(sweeper.values, f0)
            if max_freq > 4e8:
                raise_error(ValueError, f"Frequency {max_freq} for qubit {qubit.name} is beyond instrument bandwidth.")

        # is it fine to have this declaration inside the ``nshots`` QUA loop?
        f = declare(int)
        with for_(*from_array(f, sweeper.values.astype(int))):
            for pulse, f0 in zip(sweeper.pulses, freqs0):
                qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                update_frequency(qmpulse.element, f + f0)

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    def sweep_amplitude(self, sweepers, qubits, qmsequence, relaxation_time):
        from qm.qua import amp

        sweeper = sweepers[0]
        # TODO: Consider sweeping amplitude without multiplication
        if min(sweeper.values) < -2:
            raise_error(ValueError, "Amplitude sweep values are <-2 which is not supported.")
        if max(sweeper.values) > 2:
            raise_error(ValueError, "Amplitude sweep values are >2 which is not supported.")

        a = declare(fixed)
        with for_(*from_array(a, sweeper.values)):
            for pulse in sweeper.pulses:
                qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                if isinstance(qmpulse, BakedPulse):
                    qmpulse.amplitude = a
                else:
                    qmpulse.operation = qmpulse.operation * amp(a)

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    def sweep_relative_phase(self, sweepers, qubits, qmsequence, relaxation_time):
        sweeper = sweepers[0]
        relphase = declare(fixed)
        with for_(*from_array(relphase, sweeper.values / (2 * np.pi))):
            for pulse in sweeper.pulses:
                qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                qmpulse.relative_phase = relphase

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    def sweep_bias(self, sweepers, qubits, qmsequence, relaxation_time):
        from qm.qua import set_dc_offset

        sweeper = sweepers[0]
        offset0 = []
        for qubit in sweeper.qubits:
            b0 = qubit.flux.offset
            max_offset = qubit.flux.max_offset
            max_value = self.maximum_sweep_value(sweeper.values, b0)
            check_max_offset(max_value, max_offset)
            offset0.append(declare(fixed, value=b0))
        b = declare(fixed)
        with for_(*from_array(b, sweeper.values)):
            for qubit, b0 in zip(sweeper.qubits, offset0):
                with qua.if_((b + b0) >= 0.49):
                    set_dc_offset(f"flux{qubit.name}", "single", 0.49)
                with qua.elif_((b + b0) <= -0.49):
                    set_dc_offset(f"flux{qubit.name}", "single", -0.49)
                with qua.else_():
                    set_dc_offset(f"flux{qubit.name}", "single", (b + b0))

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    def sweep_start(self, sweepers, qubits, qmsequence, relaxation_time):
        sweeper = sweepers[0]
        if min(sweeper.values) < 16:
            raise_error(ValueError, "Cannot sweep start less than 16ns.")

        start = declare(int)
        values = np.array(sweeper.values) // 4
        with for_(*from_array(start, values.astype(int))):
            for pulse in sweeper.pulses:
                qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                # find all pulses that are connected to ``qmpulse`` and update their starts
                to_process = {qmpulse}
                while to_process:
                    next_qmpulse = to_process.pop()
                    to_process |= next_qmpulse.next_pulses
                    next_qmpulse.wait_time_variable = start

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    def sweep_duration(self, sweepers, qubits, qmsequence, relaxation_time):
        sweeper = sweepers[0]
        for pulse in sweeper.pulses:
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
            if isinstance(qmpulse, BakedPulse):
                values = np.array(sweeper.values).astype(int)
                qmpulse.bake(self.config, values)
            else:
                values = np.array(sweeper.values).astype(int) // 4

        dur = declare(int)

        with for_(*from_array(dur, values)):
            for pulse in sweeper.pulses:
                qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                qmpulse.swept_duration = dur
                # find all pulses that are connected to ``qmpulse`` and align them
                to_process = set(qmpulse.next_pulses)
                while to_process:
                    next_qmpulse = to_process.pop()
                    to_process |= next_qmpulse.next_pulses
                    qmpulse.elements_to_align.add(next_qmpulse.element)
                    next_qmpulse.wait_time -= qmpulse.wait_time + qmpulse.duration // 4

            self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

    SWEEPERS = {
        Parameter.frequency: sweep_frequency,
        Parameter.amplitude: sweep_amplitude,
        Parameter.relative_phase: sweep_relative_phase,
        Parameter.bias: sweep_bias,
        Parameter.start: sweep_start,
        Parameter.duration: sweep_duration,
    }

    def sweep_recursion(self, sweepers, qubits, qmsequence, relaxation_time):
        if len(sweepers) > 0:
            parameter = sweepers[0].parameter
            if parameter in self.SWEEPERS:
                self.SWEEPERS[parameter](self, sweepers, qubits, qmsequence, relaxation_time)
            else:
                raise_error(NotImplementedError, f"Sweeper for {parameter} is not implemented.")
        else:
            self.play_pulses(qmsequence, relaxation_time)
