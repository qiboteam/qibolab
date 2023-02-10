import collections

import numpy as np
from qibo.config import log, raise_error
from qm.qua import (
    align,
    assign,
    declare,
    declare_stream,
    dual_demod,
    fixed,
    for_,
    frame_rotation_2pi,
    measure,
    play,
    program,
    reset_frame,
    reset_phase,
    save,
    stream_processing,
    wait,
)
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.pulses import Pulse, PulseType, Rectangular
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter


class QMPulse:
    def __init__(self, pulse):
        self.pulse = pulse
        self.element = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.operation = pulse.serial
        self.relative_phase = pulse.relative_phase / (2 * np.pi)

        # Stores the baking object (for pulses that need 1ns resolution)
        self.baked = None
        self.baked_amplitude = None

        self.I = None
        self.Q = None
        self.shot = None
        self.I_st = None
        self.Q_st = None
        self.shots = None
        self.threshold = None
        self.cos = None
        self.sin = None

    def declare_output(self, threshold=None, iq_angle=None):
        self.I = declare(fixed)
        self.Q = declare(fixed)
        self.I_st = declare_stream()
        self.Q_st = declare_stream()

        if threshold is not None:
            # QUA variables used for single shot classification
            self.shot = declare(bool)
            self.shots = declare_stream()
            self.threshold = threshold
            self.cos = np.cos(iq_angle)
            self.sin = np.sin(iq_angle)

    def bake(self, config):
        from qualang_tools.bakery import baking

        if self.baked is not None:
            raise_error(RuntimeError, f"Bake was already called for {self.pulse}.")
        # ! Only works for flux pulses that have zero Q waveform

        # Create the different baked sequences, each one corresponding to a different truncated duration
        # for t in range(self.pulse.duration + 1):
        with baking(config, padding_method="right") as self.baked:
            # if t == 0:  # Otherwise, the baking will be empty and will not be created
            #    wf = [0.0] * 16
            # else:
            # wf = waveform[:t].tolist()
            if self.pulse.duration == 0:
                waveform = [0.0] * 16
            else:
                waveform = self.pulse.envelope_waveform_i.data.tolist()
            self.baked.add_op(self.pulse.serial, self.element, waveform)
            self.baked.play(self.pulse.serial, self.element)

            # Append the baking object in the list to call it from the QUA program
            # self.segments.append(b)


class QMSequence(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep track of readout pulses for registering outputs
        self.ro_pulses = []
        # map from qibolab pulses to QMPulses (useful when sweeping)
        self.pulse_to_qmpulse = {}

    def add(self, pulse):
        if not isinstance(pulse, Pulse):
            raise_error(TypeError, f"Pulse {pulse} has invalid type {type(pulse)}.")

        qmpulse = QMPulse(pulse)
        self.pulse_to_qmpulse[pulse.serial] = qmpulse
        if pulse.type.name == "READOUT":
            self.ro_pulses.append(qmpulse)
        super().append(qmpulse)
        return qmpulse


class QMOPX(AbstractInstrument):
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

    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        manager (:class:`qm.QuantumMachinesManager.QuantumMachinesManager`): Manager object
            used for controlling the QM OPXs.
        config (dict): Configuration dictionary required for pulse execution on the OPXs.
        time_of_flight (int): Time of flight used for hardware signal integration.
        smearing (int): Smearing used for hardware signal integration.
    """

    def __init__(self, name, address):
        # QuantumMachines manager is instantiated in ``platform.connect``
        self.name = name
        self.address = address
        self.manager = None
        self.is_connected = False

        self.time_of_flight = 0
        self.smearing = 0
        # copied from qblox runcard, not used here yet
        # hardware_avg: 1024
        # sampling_rate: 1_000_000_000
        # repetition_duration: 200_000
        # minimum_delay_between_instructions: 4

        # Default configuration for communicating with the ``QuantumMachinesManager``
        self.config = {
            "version": 1,
            "controllers": {},
            "elements": {},
            "pulses": {},
            "waveforms": {},
            "digital_waveforms": {
                "ON": {"samples": [(1, 0)]},
            },
            "integration_weights": {},
            "mixers": {},
        }

    def connect(self):
        """Connect to the QM manager."""
        host, port = self.address.split(":")
        self.manager = QuantumMachinesManager(host, int(port))

    def setup(self, qubits, relaxation_time=0, time_of_flight=0, smearing=0, **_kwargs):
        """Set general machine options and register flux elements in the ``config``.

        Flux elements should be registered on all qubits even when they are not used.

        Args:
            qubits (list): List of :class:`qibolab.platforms.abstract.Qubit`.
            relaxation_time (int): Default time to wait between shots so that the qubit relaxes to
            its ground state.
            time_of_flight (int): Time of flight used for hardware signal integration.
            smearing (int): Smearing used for hardware signal integration.
        """
        self.time_of_flight = time_of_flight
        self.smearing = smearing
        # controllers are defined when registering pulses
        for qubit in qubits.values():
            if qubit.flux:
                self.register_flux_element(qubit)

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

    @staticmethod
    def iq_imbalance(g, phi):
        """Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances

        More information here:
        https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

        Args:
            g (float): relative gain imbalance between the I & Q ports (unit-less).
                Set to 0 for no gain imbalance.
            phi (float): relative phase imbalance between the I & Q ports (radians).
                Set to 0 for no phase imbalance.
        """
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g**2) * (2 * c**2 - 1))
        return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

    def register_analog_output_controllers(self, ports, offset=0.0, filter=None):
        """Register controllers in the ``config``.

        Args:
            ports (list): List of tuples ``(conX, port)``.
            offset (float): Constant offset to be played in the given ports.
                Relevant for ports connected to flux channels.
            filter (dict): Pulse shape filters. Relevant for ports connected to flux channels.
                QM syntax should be followed for the filters.
        """
        if abs(offset) > 0.2:
            raise_error(ValueError, f"DC offset for Quantum Machines cannot exceed 0.1V but is {offset}.")

        controllers = self.config["controllers"]
        for con, port in ports:
            if con not in controllers:
                controllers[con] = {"analog_outputs": {}}
            controllers[con]["analog_outputs"][port] = {"offset": offset}
            if filter is not None:
                controllers[con]["analog_outputs"][port]["filter"] = filter

    def register_drive_element(self, qubit, intermediate_frequency=0):
        """Register qubit drive elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"drive{qubit.name}" not in self.config["elements"]:
            # register drive controllers
            self.register_analog_output_controllers(qubit.drive.ports)
            # register element
            lo_frequency = int(qubit.drive.local_oscillator.frequency)
            self.config["elements"][f"drive{qubit.name}"] = {
                "mixInputs": {
                    "I": qubit.drive.ports[0],
                    "Q": qubit.drive.ports[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_drive{qubit.name}",
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
            drive_g = qubit.mixer_drive_g
            drive_phi = qubit.mixer_drive_phi
            self.config["mixers"][f"mixer_drive{qubit.name}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(drive_g, drive_phi),
                }
            ]
        else:
            self.config["elements"][f"drive{qubit.name}"]["intermediate_frequency"] = intermediate_frequency
            self.config["mixers"][f"mixer_drive{qubit.name}"][0]["intermediate_frequency"] = intermediate_frequency

    def register_readout_element(self, qubit, intermediate_frequency=0):
        """Register resonator elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"readout{qubit.name}" not in self.config["elements"]:
            # register readout controllers
            self.register_analog_output_controllers(qubit.readout.ports)
            # register feedback controllers
            controllers = self.config["controllers"]
            for con, port in qubit.feedback.ports:
                if con not in controllers:
                    controllers[con] = {
                        "analog_outputs": {},
                        "digital_outputs": {
                            1: {},
                        },
                        "analog_inputs": {},
                    }
                if "digital_outputs" not in controllers[con]:
                    controllers[con]["digital_outputs"] = {
                        1: {},
                    }
                if "analog_inputs" not in controllers[con]:
                    controllers[con]["analog_inputs"] = {}
                controllers[con]["analog_inputs"][port] = {"offset": 0.0, "gain_db": 0}

            # register element
            lo_frequency = int(qubit.readout.local_oscillator.frequency)
            self.config["elements"][f"readout{qubit.name}"] = {
                "mixInputs": {
                    "I": qubit.readout.ports[0],
                    "Q": qubit.readout.ports[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_readout{qubit.name}",
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
                "outputs": {
                    "out1": qubit.feedback.ports[0],
                    "out2": qubit.feedback.ports[1],
                },
                "time_of_flight": self.time_of_flight,
                "smearing": self.smearing,
            }
            readout_g = qubit.mixer_readout_g
            readout_phi = qubit.mixer_readout_phi
            self.config["mixers"][f"mixer_readout{qubit.name}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(readout_g, readout_phi),
                }
            ]
        else:
            self.config["elements"][f"readout{qubit.name}"]["intermediate_frequency"] = intermediate_frequency
            self.config["mixers"][f"mixer_readout{qubit.name}"][0]["intermediate_frequency"] = intermediate_frequency

    def register_flux_element(self, qubit, intermediate_frequency=0):
        """Register qubit flux elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"flux{qubit.name}" not in self.config["elements"]:
            # register controller
            self.register_analog_output_controllers(qubit.flux.ports, qubit.flux.offset, qubit.flux.filter)
            # register element
            self.config["elements"][f"flux{qubit.name}"] = {
                "singleInput": {
                    "port": qubit.flux.ports[0],
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
        else:
            self.config["elements"][f"flux{qubit.name}"]["intermediate_frequency"] = intermediate_frequency

    def register_pulse(self, qubit, pulse):
        """Registers pulse, waveforms and integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit that the pulse acts on.
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to register.

        Returns:
            element (str): Name of the element this pulse will be played on.
                Elements are a part of the QM config and are generated during
                instantiation of the Qubit objects. They are named as
                "drive0", "drive1", "flux0", "readout0", ...
        """
        pulses = self.config["pulses"]
        mixers = self.config["mixers"]
        if pulse.serial not in pulses:
            if pulse.type.name == "DRIVE":
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {"I": serial_i, "Q": serial_q},
                }
                # register drive element (if it does not already exist)
                if_frequency = pulse.frequency - int(qubit.drive.local_oscillator.frequency)
                self.register_drive_element(qubit, if_frequency)
                # register flux element (if available)
                if qubit.flux:
                    self.register_flux_element(qubit)
                # register drive pulse in elements
                self.config["elements"][f"drive{qubit.name}"]["operations"][pulse.serial] = pulse.serial

            elif pulse.type.name == "FLUX":
                serial = self.register_waveform(pulse)
                pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": serial,
                    },
                }
                # register flux element (if it does not already exist)
                self.register_flux_element(qubit, pulse.frequency)
                # register flux pulse in elements
                self.config["elements"][f"flux{qubit.name}"]["operations"][pulse.serial] = pulse.serial

            elif pulse.type.name == "READOUT":
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.register_integration_weights(qubit, pulse.duration)
                pulses[pulse.serial] = {
                    "operation": "measurement",
                    "length": pulse.duration,
                    "waveforms": {
                        "I": serial_i,
                        "Q": serial_q,
                    },
                    "integration_weights": {
                        "cos": f"cosine_weights{qubit.name}",
                        "sin": f"sine_weights{qubit.name}",
                        "minus_sin": f"minus_sine_weights{qubit.name}",
                    },
                    "digital_marker": "ON",
                }
                # register readout element (if it does not already exist)
                if_frequency = pulse.frequency - int(qubit.readout.local_oscillator.frequency)
                self.register_readout_element(qubit, if_frequency)
                # register flux element (if available)
                if qubit.flux:
                    self.register_flux_element(qubit)
                # register readout pulse in elements
                self.config["elements"][f"readout{qubit.name}"]["operations"][pulse.serial] = pulse.serial

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

    def register_waveform(self, pulse, mode="i"):
        """Registers waveforms in QM config.

        QM supports two kinds of waveforms, examples:
            "zero_wf": {"type": "constant", "sample": 0.0}
            "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()}

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to read the waveform from.
            mode (str): "i" or "q" specifying which channel the waveform will be played.

        Returns:
            serial (str): String with a serialization of the waveform.
                Used as key to identify the waveform in the config.
        """
        waveforms = self.config["waveforms"]
        # Maybe need to force zero q waveforms
        # if pulse.type.name == "READOUT" and mode == "q":
        #    serial = "zero_wf"
        #    if serial not in waveforms:
        #        waveforms[serial] = {"type": "constant", "sample": 0.0}
        if isinstance(pulse.shape, Rectangular):
            serial = f"constant_wf{pulse.amplitude}"
            if serial not in waveforms:
                waveforms[serial] = {"type": "constant", "sample": pulse.amplitude}
        else:
            waveform = getattr(pulse, f"envelope_waveform_{mode}")
            serial = waveform.serial
            if serial not in waveforms:
                waveforms[serial] = {"type": "arbitrary", "samples": waveform.data.tolist()}
        return serial

    def register_integration_weights(self, qubit, readout_len):
        """Registers integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.quantum_machines.Qubit`): Qubit
                object that the integration weights will be used for.
            readout_len (int): Duration of the readout pulse in ns.
        """
        rotation_angle = qubit.rotation_angle
        self.config["integration_weights"].update(
            {
                f"cosine_weights{qubit.name}": {
                    "cosine": [(np.cos(rotation_angle), readout_len)],
                    "sine": [(-np.sin(rotation_angle), readout_len)],
                },
                f"sine_weights{qubit.name}": {
                    "cosine": [(np.sin(rotation_angle), readout_len)],
                    "sine": [(np.cos(rotation_angle), readout_len)],
                },
                f"minus_sine_weights{qubit.name}": {
                    "cosine": [(-np.sin(rotation_angle), readout_len)],
                    "sine": [(-np.cos(rotation_angle), readout_len)],
                },
            }
        )

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.

        Returns:
            TODO
        """
        machine = self.manager.open_qm(self.config)

        # for debugging only
        from qm import generate_qua_script

        with open("qua_script.txt", "w") as file:
            file.write(generate_qua_script(program, self.config))

        return machine.execute(program)

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
        align()
        clock = collections.defaultdict(int)
        for qmpulse in qmsequence:
            pulse = qmpulse.pulse
            wait_time = pulse.start - clock[qmpulse.element]
            if wait_time >= 12:
                wait(wait_time // 4 + 1, qmpulse.element)
            clock[qmpulse.element] += pulse.duration
            if pulse.type.name == "READOUT":
                measure(
                    qmpulse.operation,
                    qmpulse.element,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", qmpulse.I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", qmpulse.Q),
                )
                if qmpulse.threshold is not None:
                    assign(qmpulse.shot, qmpulse.I * qmpulse.cos - qmpulse.Q * qmpulse.sin > qmpulse.threshold)
            else:
                if not isinstance(qmpulse.relative_phase, float) or qmpulse.relative_phase != 0:
                    frame_rotation_2pi(qmpulse.relative_phase, qmpulse.element)
                    needs_reset = True
                if qmpulse.baked is not None:
                    if qmpulse.baked_amplitude is not None:
                        qmpulse.baked.run(amp_array=[(qmpulse.element, qmpulse.baked_amplitude)])
                    else:
                        qmpulse.baked.run()
                else:
                    play(qmpulse.operation, qmpulse.element)
                if needs_reset:
                    reset_frame(qmpulse.element)

        # for Rabi-length?
        if relaxation_time > 0:
            wait(relaxation_time // 4, *clock.keys())

        # Save data to the stream processing
        for qmpulse in qmsequence.ro_pulses:
            save(qmpulse.I, qmpulse.I_st)
            save(qmpulse.Q, qmpulse.Q_st)
            if qmpulse.threshold is not None:
                save(qmpulse.shot, qmpulse.shots)

    @staticmethod
    def fetch_results(result, ro_pulses):
        """Fetches results from an executed experiment."""
        # TODO: Update result asynchronously instead of waiting
        # for all values, in order to allow live plotting
        # import time
        # for _ in range(5):
        #    handles.is_processing()
        #    time.sleep(1)
        handles = result.result_handles
        handles.wait_for_all_values()
        results = {}
        for pulse in ro_pulses:
            serial = pulse.serial
            ires = handles.get(f"{serial}_I").fetch_all()
            qres = handles.get(f"{serial}_Q").fetch_all()
            if f"{serial}_shots" in handles:
                shots = handles.get(f"{serial}_shots").fetch_all().astype(int)
            else:
                shots = None
            results[pulse.qubit] = results[serial] = ExecutionResults.from_components(ires, qres, shots)
        return results

    def play(self, qubits, sequence, nshots, relaxation_time):
        """Plays an arbitrary pulse sequence using QUA program.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects
                passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            nshots (int): Number of repetitions (shots) of the experiment.
            relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in ns.
        """
        if not sequence:
            return {}
        # Current driver cannot play overlapping pulses on drive and flux channels
        # If we want to play overlapping pulses we need to define different elements on the same ports
        # like we do for readout multiplex
        qmsequence = QMSequence()
        for pulse in sequence:
            qmpulse = qmsequence.add(pulse)
            if pulse.duration % 4 or pulse.duration < 16:
                if pulse.type.name != "FLUX":
                    raise_error(NotImplementedError, "1ns resolution is available for flux pulses only.")
                # register flux element (if it does not already exist)
                self.register_flux_element(qubits[pulse.qubit], pulse.frequency)
                qmpulse.bake(self.config)
            else:
                self.register_pulse(qubits[pulse.qubit], pulse)

        # play pulses using QUA
        with program() as experiment:
            n = declare(int)
            for qmpulse in qmsequence.ro_pulses:
                threshold = qubits[qmpulse.pulse.qubit].threshold
                iq_angle = qubits[qmpulse.pulse.qubit].iq_angle
                qmpulse.declare_output(threshold, iq_angle)

            with for_(n, 0, n < nshots, n + 1):
                self.play_pulses(qmsequence, relaxation_time)

            with stream_processing():
                # I_st.average().save("I")
                # Q_st.average().save("Q")
                # n_st.buffer().save_all("n")
                for qmpulse in qmsequence.ro_pulses:
                    serial = qmpulse.pulse.serial
                    qmpulse.I_st.buffer(nshots).save(f"{serial}_I")
                    qmpulse.Q_st.buffer(nshots).save(f"{serial}_Q")
                    if qmpulse.threshold is not None:
                        qmpulse.shots.buffer(nshots).save(f"{serial}_shots")

        result = self.execute_program(experiment)
        return self.fetch_results(result, sequence.ro_pulses)

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        qmsequence = QMSequence()
        for pulse in sequence:
            qmpulse = qmsequence.add(pulse)
            if pulse.duration % 4 or pulse.duration < 16:
                if pulse.type.name != "FLUX":
                    raise_error(NotImplementedError, "1ns resolution is available for flux pulses only.")
                # register flux element (if it does not already exist)
                self.register_flux_element(qubits[pulse.qubit], pulse.frequency)
                qmpulse.bake(self.config)
            else:
                self.register_pulse(qubits[pulse.qubit], pulse)

        # play pulses using QUA
        with program() as experiment:
            n = declare(int)
            for qmpulse in qmsequence.ro_pulses:
                if average:
                    # not calculating single shots when averaging
                    # during sweep so we do not pass ``threshold`` here
                    qmpulse.declare_output()
                else:
                    threshold = qubits[qmpulse.pulse.qubit].threshold
                    iq_angle = qubits[qmpulse.pulse.qubit].iq_angle
                    qmpulse.declare_output(threshold, iq_angle)

            with for_(n, 0, n < nshots, n + 1):
                self.sweep_recursion(list(sweepers), qubits, qmsequence, relaxation_time)

            with stream_processing():
                for qmpulse in qmsequence.ro_pulses:
                    Ist_temp = qmpulse.I_st
                    Qst_temp = qmpulse.Q_st
                    if not average and qmpulse.threshold is not None:
                        shots_temp = qmpulse.shots
                    for sweeper in reversed(sweepers):
                        Ist_temp = Ist_temp.buffer(len(sweeper.values))
                        Qst_temp = Qst_temp.buffer(len(sweeper.values))
                        if not average and qmpulse.threshold is not None:
                            shots_temp = shots_temp.buffer(len(sweeper.values))
                    serial = qmpulse.pulse.serial
                    if average:
                        Ist_temp.average().save(f"{serial}_I")
                        Qst_temp.average().save(f"{serial}_Q")
                    else:
                        Ist_temp.buffer(nshots).save(f"{serial}_I")
                        Qst_temp.buffer(nshots).save(f"{serial}_Q")
                        if qmpulse.threshold is not None:
                            shots_temp.buffer(nshots).save(f"{serial}_shots")

        result = self.execute_program(experiment)
        return self.fetch_results(result, sequence.ro_pulses)

    def sweep_recursion(self, sweepers, qubits, qmsequence, relaxation_time):
        from qualang_tools.loops import from_array

        sweeper = sweepers[0]
        if sweeper.pulses is not None:
            if sweeper.parameter is Parameter.frequency:
                from qm.qua import update_frequency

                freqs0 = []
                for pulse in sweeper.pulses:
                    qubit = qubits[pulse.qubit]
                    if pulse.type is PulseType.DRIVE:
                        lo_frequency = int(qubit.drive.local_oscillator.frequency)
                    elif pulse.type is PulseType.READOUT:
                        lo_frequency = int(qubit.readout.local_oscillator.frequency)
                    else:
                        raise_error(NotImplementedError, f"Cannot sweep frequency of pulse of type {pulse.type}.")
                    # convert to IF frequency for readout and drive pulses
                    freqs0.append(declare(int, value=int(pulse.frequency - lo_frequency)))

                # is it fine to have this declaration inside the ``nshots`` QUA loop?
                f = declare(int)
                with for_(*from_array(f, sweeper.values.astype(int))):
                    for pulse, f0 in zip(sweeper.pulses, freqs0):
                        qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                        update_frequency(qmpulse.element, f + f0)
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
                    else:
                        self.play_pulses(qmsequence)
                    if relaxation_time > 0:
                        wait(relaxation_time // 4)

            elif sweeper.parameter is Parameter.amplitude:
                from qm.qua import amp

                # TODO: It should be -2 < amp(a) < 2 otherwise the we get weird results
                # without an error. Amplitude should be fixed to allow arbitrary values
                # in qibocal

                a = declare(fixed)
                with for_(*from_array(a, sweeper.values)):
                    for pulse in sweeper.pulses:
                        qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                        if qmpulse.baked is None:
                            qmpulse.operation = qmpulse.operation * amp(a)
                        else:
                            qmpulse.baked_amplitude = a
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
                    else:
                        self.play_pulses(qmsequence)
                    if relaxation_time > 0:
                        wait(relaxation_time // 4)

            elif sweeper.parameter is Parameter.relative_phase:
                relphase = declare(fixed)
                with for_(*from_array(relphase, sweeper.values / (2 * np.pi))):
                    for pulse in sweeper.pulses:
                        qmpulse = qmsequence.pulse_to_qmpulse[pulse.serial]
                        qmpulse.relative_phase = relphase
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
                    else:
                        self.play_pulses(qmsequence)
                    if relaxation_time > 0:
                        wait(relaxation_time // 4)

            else:
                raise_error(NotImplementedError, "Sweeper configuration not implemented.")

        elif sweeper.qubits is not None:
            if sweeper.parameter is Parameter.bias:
                from qm.qua import set_dc_offset

                bias0 = [declare(fixed, value=qubits[q].flux.offset) for q in sweeper.qubits]
                b = declare(fixed)
                with for_(*from_array(b, sweeper.values)):
                    for q, b0 in zip(sweeper.qubits, bias0):
                        set_dc_offset(f"flux{q}", "single", b + b0)
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
                    else:
                        self.play_pulses(qmsequence)
                    if relaxation_time > 0:
                        elements = (f"flux{q}" for q in sweeper.qubits)
                        wait(relaxation_time // 4, *elements)

            else:
                raise_error(NotImplementedError, "Sweeper configuration not implemented.")

        else:
            raise_error(NotImplementedError, "Sweeper configuration not implemented.")
