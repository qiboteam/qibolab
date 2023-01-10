import collections
from types import SimpleNamespace

import numpy as np
from qibo.config import log, raise_error
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.result import ExecutionResult


class QMOPX(AbstractInstrument):
    """Instrument object for controlling Quantum Machines (QM) OPX controllers.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language. The ``config`` file is generated in parts in the following places:
        - controllers are registered in ``__init__``,
        -  are registered in each ``Qubit`` object,
        - elements (qubits, resonators and flux), pulses (including waveforms
          and integration weights) are registered in the ``register_*`` methods.
    The QUA program for executing an arbitrary qibolab ``PulseSequence`` is written in
    ``execute_program`` which is called by ``play``.

    Args:
        name (str): Name of the instrument instance.
        address (str): IP address and port for connecting to the OPX instruments.

    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        manager (:class:`qm.QuantumMachinesManager.QuantumMachinesManager`): Manager object
            used for controlling the QM OPXs.
        config (dict): Configuration dictionary required for pulse execution on the OPXs.
        time_of_flight (optional,int): Time of flight used for hardware signal integration.
        smearing (optional,int): Smearing used for hardware signal integration.
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
        # Defines which controllers and ports are used in the lab (HARDCODED)
        # TODO: Generate ``config`` controllers in ``setup`` by looking at the channels
        self.config = {
            "version": 1,
            "controllers": {
                "con1": {
                    "analog_outputs": {
                        1: {"offset": 0.0},
                        2: {"offset": 0.0},
                        3: {"offset": 0.0},
                        4: {"offset": 0.0},
                        5: {"offset": 0.0},
                        6: {"offset": 0.0},
                        7: {"offset": 0.0},
                        8: {"offset": 0.0},
                        9: {"offset": 0.0},
                        10: {"offset": 0.0},
                    },
                    "digital_outputs": {
                        1: {},
                    },
                    "analog_inputs": {
                        1: {"offset": 0.0, "gain_db": 0},
                        2: {"offset": 0.0, "gain_db": 0},
                    },
                },
                "con2": {
                    "analog_outputs": {
                        1: {"offset": 0.0, "filter": {"feedforward": [], "feedback": []}},
                        2: {"offset": 0.0, "filter": {"feedforward": [], "feedback": []}},
                        3: {"offset": 0.0, "filter": {"feedforward": [], "feedback": []}},
                        4: {"offset": 0.0, "filter": {"feedforward": [], "feedback": []}},
                        5: {"offset": 0.0, "filter": {"feedforward": [], "feedback": []}},
                        9: {"offset": 0.0},
                        10: {"offset": 0.0},
                    },
                },
                "con3": {
                    "analog_outputs": {
                        1: {"offset": 0.0},
                        2: {"offset": 0.0},
                    },
                },
            },
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
        host, port = self.address.split(":")
        self.manager = QuantumMachinesManager(host, int(port))

    def setup(self, qubits, time_of_flight=0, smearing=0):
        self.time_of_flight = time_of_flight
        self.smearing = smearing
        # TODO: Use ``qubits`` to define controllers here

    def start(self):
        # TODO: Start the OPX flux offsets?
        pass

    def stop(self):
        if self.is_connected:
            self.manager.close_all_quantum_machines()

    def disconnect(self):
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

    def register_drive_element(self, qubit, intermediate_frequency=0):
        """Register qubit drive elements in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"drive{qubit}" not in self.config["elements"]:
            lo_frequency = qubit.drive.lo_frequency
            self.config["elements"][f"drive{qubit}"] = {
                "mixInputs": {
                    "I": qubit.drive.ports[0],
                    "Q": qubit.drive.ports[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_drive{qubit}",
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
            drive_g = qubit.characterization.mixer_drive_g
            drive_phi = qubit.characterization.mixer_drive_phi
            self.config["mixers"][f"mixer_drive{qubit}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(drive_g, drive_phi),
                }
            ]
        else:
            current_if = self.config["elements"][f"drive{qubit}"]["intermediate_frequency"]
            if current_if != intermediate_frequency:
                raise_error(NotImplementedError, f"Changing intermediate frequency for qubit {qubit}.")

    def register_readout_element(self, qubit, intermediate_frequency=0):
        """Register resonator elements in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"readout{qubit}" not in self.config["elements"]:
            lo_frequency = qubit.readout.lo_frequency
            self.config["elements"][f"readout{qubit}"] = {
                "mixInputs": {
                    "I": qubit.readout.ports[0],
                    "Q": qubit.readout.ports[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_readout{qubit}",
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
            readout_g = qubit.characterization.mixer_readout_g
            readout_phi = qubit.characterization.mixer_readout_phi
            self.config["mixers"][f"mixer_readout{qubit}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(readout_g, readout_phi),
                }
            ]
        else:
            current_if = self.config["elements"][f"readout{qubit}"]["intermediate_frequency"]
            if current_if != intermediate_frequency:
                raise_error(NotImplementedError, f"Changing intermediate frequency for qubit {qubit}.")

    def register_flux_element(self, qubit, intermediate_frequency=0):
        """Register qubit flux elements in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"flux{qubit}" not in self.config["elements"]:
            self.config["elements"][f"flux{qubit}"] = {
                "singleInput": {
                    "port": qubit.flux.ports[0],
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
        else:
            current_if = self.config["elements"][f"readout{qubit}"]["intermediate_frequency"]
            if current_if != intermediate_frequency:
                raise_error(NotImplementedError, f"Changing intermediate frequency for qubit {qubit}.")

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
                if_frequency = pulse.frequency - qubit.drive.lo_frequency
                self.register_drive_element(qubit, if_frequency)
                # register drive pulse in elements
                self.config["elements"][f"drive{qubit}"]["operations"][pulse.serial] = pulse.serial

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
                self.config["elements"][f"flux{qubit}"]["operations"][pulse.serial] = pulse.serial

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
                        "cos": f"cosine_weights{qubit}",
                        "sin": f"sine_weights{qubit}",
                        "minus_sin": f"minus_sine_weights{qubit}",
                    },
                    "digital_marker": "ON",
                }
                # register readout element (if it does not already exist)
                if_frequency = pulse.frequency - qubit.readout.lo_frequency
                self.register_readout_element(qubit, if_frequency)
                # register readout pulse in elements
                self.config["elements"][f"readout{qubit}"]["operations"][pulse.serial] = pulse.serial

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

        return f"{pulse.type.name.lower()}{str(qubit)}"

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
        from qibolab.pulses import Rectangular

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
        rotation_angle = qubit.characterization.rotation_angle
        self.config["integration_weights"] = {
            f"cosine_weights{qubit}": {
                "cosine": [(np.cos(rotation_angle), readout_len)],
                "sine": [(-np.sin(rotation_angle), readout_len)],
            },
            f"sine_weights{qubit}": {
                "cosine": [(np.sin(rotation_angle), readout_len)],
                "sine": [(np.cos(rotation_angle), readout_len)],
            },
            f"minus_sine_weights{qubit}": {
                "cosine": [(-np.sin(rotation_angle), readout_len)],
                "sine": [(-np.cos(rotation_angle), readout_len)],
            },
        }

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.

        Returns:
            TODO
        """
        machine = self.manager.open_qm(self.config)

        from qm import generate_qua_script

        print()
        print(generate_qua_script(program, self.config))
        print()
        return machine.execute(program)

    @staticmethod
    def play_pulses(qmsequence, outputs):
        """Part of QUA program that plays an arbitrary pulse sequence.

        Should be used inside a ``program()`` context.

        Args:
            qmsequence (list): Pulse sequence containing QM specific pulses (``qmpulse``).
                These pulses are defined in :meth:`qibolab.instruments.qm.QMOPX.play` and
                hold attributes for the ``target`` and ``operation`` that corresponds to
                each pulse, as defined in the QM config.
            outputs (dict): Dictionary with the QUA variables and streams that will
                save the results for each readout pulse.
        """
        from qm.qua import align, dual_demod, measure, play, save, wait

        # TODO: Fix phases
        # TODO: Handle pulses that run on the same element simultaneously (multiplex?)
        # register pulses in Quantum Machines config

        clock = collections.defaultdict(int)
        for qmpulse in qmsequence:
            pulse = qmpulse.pulse
            wait_time = pulse.start - clock[pulse.qubit]
            if wait_time > 0:
                wait(wait_time // 4, qmpulse.target)
            clock[pulse.qubit] += pulse.duration
            if pulse.type.name == "READOUT":
                # align("qubit", "resonator")
                align()
                measure(
                    qmpulse.operation,
                    qmpulse.target,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", outputs[pulse.serial].I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", outputs[pulse.serial].Q),
                )
            else:
                play(qmpulse.operation, qmpulse.target)
        # Save data to the stream processing
        for output in outputs.values():
            save(output.I, output.I_st)
            save(output.Q, output.Q_st)

    @staticmethod
    def fetch_results(result, readout_serials):
        """Fetches results from an executed experiment."""
        # TODO: Fetch in real time during sweeping, to allow live plotting
        handles = result.result_handles
        handles.wait_for_all_values()
        results = {}
        for serial in readout_serials:
            ires = handles.get(f"{serial}_I").fetch_all()
            qres = handles.get(f"{serial}_Q").fetch_all()
            results[serial] = ExecutionResult(ires, qres)
        return results

    def play(self, qubits, sequence, nshots=1024):
        """Plays an arbitrary pulse sequence using QUA program.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects
                passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            nshots (int): Number of repetitions (shots) of the experiment.
        """
        from qm.qua import (
            declare,
            declare_stream,
            fixed,
            for_,
            program,
            stream_processing,
        )

        qmsequence = [
            SimpleNamespace(pulse=pulse, target=self.register_pulse(qubits[pulse.qubit], pulse), operation=pulse.serial)
            for pulse in sequence
        ]

        # play pulses using QUA
        with program() as experiment:
            n = declare(int)
            outputs = {
                pulse.serial: SimpleNamespace(
                    I=declare(fixed),
                    Q=declare(fixed),
                    I_st=declare_stream(),
                    Q_st=declare_stream(),
                )
                for pulse in sequence.ro_pulses
            }
            with for_(n, 0, n < nshots, n + 1):
                self.play_pulses(qmsequence, outputs)

            with stream_processing():
                # I_st.average().save("I")
                # Q_st.average().save("Q")
                # n_st.buffer().save_all("n")
                for serial, output in outputs.items():
                    output.I_st.buffer(nshots).save(f"{serial}_I")
                    output.Q_st.buffer(nshots).save(f"{serial}_Q")

        result = self.execute_program(experiment)
        return self.fetch_results(result, outputs.keys())

    def sweep_recursion(self, sweepers, qubits, qmpulses, qmsequence, outputs):
        from qm.qua import declare, for_, wait
        from qualang_tools.loops import from_array

        sweeper = sweepers[0]
        if sweeper.pulse is not None:
            if sweeper.parameter == "frequency":
                from qm.qua import update_frequency

                if sweeper.pulse_type in ("readout", "drive"):
                    # convert to IF frequency for readout and drive pulses
                    qubit = qubits[sweeper.pulse.qubit]
                    values = sweeper.values - getattr(qubit, sweeper.pulse_type).lo_frequency
                else:
                    values = sweeper.values

                # is it fine to have this declaration inside the ``nshots`` QUA loop?
                f = declare(int)
                qmpulse = qmpulses[sweeper.pulse.serial]
                with for_(*from_array(f, values.astype(int))):
                    update_frequency(qmpulse.target, f)
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmpulses, qmsequence, outputs)
                    else:
                        self.play_pulses(qmsequence, outputs)
                    if sweeper.wait_time > 0:
                        wait(sweeper.wait_time // 4, qmpulse.target)

            elif sweeper.parameter == "amplitude":
                from qm.qua import amp, fixed

                qmpulse = qmpulses[sweeper.pulse.serial]
                a = declare(fixed)
                with for_(*from_array(a, sweeper.values)):
                    qmpulse.operation = qmpulse.operation * amp(a)
                    if len(sweepers) > 1:
                        self.sweep_recursion(sweepers[1:], qubits, qmpulses, qmsequence, outputs)
                    else:
                        self.play_pulses(qmsequence, outputs)
                    if sweeper.wait_time > 0:
                        wait(sweeper.wait_time // 4, qmpulse.target)

            else:
                raise_error(NotImplementedError)

        else:
            raise_error(NotImplementedError)

    def sweep(self, qubits, sequence, *sweepers, nshots=1024):
        from qm.qua import (
            align,
            declare,
            declare_stream,
            fixed,
            for_,
            program,
            stream_processing,
        )

        qmsequence, qmpulses = [], {}
        for pulse in sequence:
            qmpulse = SimpleNamespace(
                pulse=pulse, target=self.register_pulse(qubits[pulse.qubit], pulse), operation=pulse.serial
            )
            qmsequence.append(qmpulse)
            qmpulses[pulse.serial] = qmpulse

        # play pulses using QUA
        with program() as experiment:
            n = declare(int)
            outputs = {
                pulse.serial: SimpleNamespace(
                    I=declare(fixed),
                    Q=declare(fixed),
                    I_st=declare_stream(),
                    Q_st=declare_stream(),
                )
                for pulse in sequence.ro_pulses
            }
            with for_(n, 0, n < nshots, n + 1):
                self.sweep_recursion(list(sweepers), qubits, qmpulses, qmsequence, outputs)
            align()

            with stream_processing():
                for serial, output in outputs.items():
                    Ist_temp = output.I_st
                    Qst_temp = output.Q_st
                    for sweeper in reversed(sweepers):
                        if sweeper.pulse.serial == serial:
                            Ist_temp = Ist_temp.buffer(len(sweeper.values))
                            Qst_temp = Qst_temp.buffer(len(sweeper.values))
                    Ist_temp.average().save(f"{serial}_I")
                    Qst_temp.average().save(f"{serial}_Q")

        # TODO: Update result asynchronously instead of waiting for all values
        # import time
        # for _ in range(5):
        #    print(handles.is_processing())
        #    time.sleep(1)
        result = self.execute_program(experiment)
        return self.fetch_results(result, outputs.keys())
