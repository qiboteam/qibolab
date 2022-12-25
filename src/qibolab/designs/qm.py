import itertools

import numpy as np
import yaml
from qibo.config import log, raise_error
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platforms.utils import Channel


class QMRSDesign:
    """Instrument design for Quantum Machines (QM) OPXs and Rohde Schwarz local oscillators.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language. The ``config`` file is generated in parts in the following places:
        - controllers are registered in ``__init__``,
        - elements (qubits, resonators and flux) are registered in each ``Qubit`` object,
        - pulses (including waveforms and integration weights) are registered in the
          ``register_*`` methods of the platform.
    The QUA program for executing an arbitrary qibolab ``PulseSequence`` is written in
    ``execute_pulse_sequence``.

    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        nqubits (int): Number of qubits in the chip.
        resonator_type (str): Type of the resonators (2D or 3D) used for qubit state readout.
        topology (list): Topology of the chip.
        settings (dict): Raw runcard loaded in a dictionary.
        manager (:class:`qm.QuantumMachinesManager.QuantumMachinesManager`): Manager object
            used for controlling the QM OPXs.
        config (dict): Configuration dictionary required for pulse execution on the OPXs.
        local_oscillators (dict): Dictionary mapping LO names to the corresponding
            instrument objects.
        qubits (list): List of :class:`qibolab.platforms.quantum_machines.Qubit` objects
            for each qubit to be controlled.
    """

    def __init__(self, address="192.168.0.1:80"):
        # QuantumMachines manager is instantiated in ``platform.connect``
        self.manager = None
        self.is_connected = False

        # Configuration values for QM (HARDCODED)
        self.address = address
        # relevant only for readout and used for hardware signal integration:
        # time_of_flight (optional,int): Time of flight associated with the channel.
        self.time_of_flight = 280
        # smearing (optional,int): Time of flight associated with the channel.
        self.smearing = 0
        # copied from qblox runcard, not used here yet
        # hardware_avg: 1024
        # sampling_rate: 1_000_000_000
        # repetition_duration: 200_000
        # minimum_delay_between_instructions: 4

        # Default configuration for communicating with the ``QuantumMachinesManager``
        # Defines which controllers and ports are used in the lab (HARDCODED)
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
        # Map controllers to qubit channels (HARDCODED)
        # readout
        Channel("L3-25_a").ports = [("con1", 9), ("con1", 10)]
        Channel("L3-25_b").ports = [("con2", 9), ("con2", 10)]
        self.readout_channels = [Channel("L3-25_a"), Channel("L3-25_b")]
        # feedback
        Channel("L2-5").ports = [("con1", 1), ("con1", 2)]
        self.feedback_channel = Channel("L2-5")
        # drive
        Channel("L3-11").ports = [("con1", 1), ("con1", 2)]
        Channel("L3-12").ports = [("con1", 3), ("con1", 4)]
        Channel("L3-13").ports = [("con1", 5), ("con1", 6)]
        Channel("L3-14").ports = [("con1", 7), ("con1", 8)]
        Channel("L3-15").ports = [("con3", 1), ("con3", 2)]
        self.drive_channels = [Channel(f"L3-{i}") for i in range(11, 16)]
        # flux
        Channel("L4-1").ports = [("con2", 1)]
        Channel("L4-2").ports = [("con2", 2)]
        Channel("L4-3").ports = [("con2", 3)]
        Channel("L4-4").ports = [("con2", 4)]
        Channel("L4-5").ports = [("con2", 5)]
        self.flux_channels = [Channel(f"L4-{i}") for i in range(1, 6)]

        # Instantiate local oscillators (HARDCODED)
        self.local_oscillators = [
            SGS100A("lo_readout_a", "192.168.0.39"),
            SGS100A("lo_readout_b", "192.168.0.31"),
            # FIXME: Temporarily disable the drive LOs since we are not using them
            SGS100A("lo_drive_low", "192.168.0.32"),
            SGS100A("lo_drive_mid", "192.168.0.33"),
            SGS100A("lo_drive_high", "192.168.0.34"),
        ]

        # Map LOs to channels
        Channel("L3-25_a").local_oscillator = self.local_oscillators[0]
        Channel("L3-25_b").local_oscillator = self.local_oscillators[1]
        Channel("L3-15").local_oscillator = self.local_oscillators[2]
        Channel("L3-11").local_oscillator = self.local_oscillators[2]
        Channel("L3-12").local_oscillator = self.local_oscillators[3]
        Channel("L3-13").local_oscillator = self.local_oscillators[4]
        Channel("L3-14").local_oscillator = self.local_oscillators[4]

        # Set default LO parameters in the channel
        Channel("L3-25_a").lo_frequency = 7_300_000_000
        Channel("L3-25_b").lo_frequency = 7_850_000_000
        # Channel("L3-25_a").lo_frequency = 7_850_000_000
        # Channel("L3-25_b").lo_frequency = 7_300_000_000
        Channel("L3-15").lo_frequency = 4_700_000_000
        Channel("L3-11").lo_frequency = 4_700_000_000
        Channel("L3-12").lo_frequency = 5_600_000_000
        Channel("L3-13").lo_frequency = 6_500_000_000
        Channel("L3-14").lo_frequency = 6_500_000_000

        Channel("L3-25_a").lo_power = 15.0
        Channel("L3-25_b").lo_power = 18.0
        Channel("L3-15").lo_power = 16.0
        Channel("L3-11").lo_power = 16.0
        Channel("L3-12").lo_power = 16.0
        Channel("L3-13").lo_power = 16.0
        Channel("L3-14").lo_power = 16.0

    def connect(self):
        """Connects to all instruments.

        Args:
            host (optional, str): Optional can be given to connect to a cloud simulator
                instead of the actual instruments. If the ``host`` is not given,
                this will connect to the physical OPXs and LOs using the addresses
                given in the runcard.
        """
        host, port = self.address.split(":")
        self.manager = QuantumMachinesManager(host, int(port))
        if not self.is_connected:
            for lo in self.local_oscillators:
                try:
                    log.info(f"Connecting to instrument {lo}.")
                    lo.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {lo} instruments. Error captured: '{exception}'",
                    )
            self.is_connected = True

    def setup(self, qubits):
        # register qubit elements in the QM config
        for qubit in qubits:
            self.register_element(qubit)

        # set LO frequencies
        for channel in itertools.chain(self.readout_channels, self.drive_channels):
            if channel.local_oscillator is not None:
                # set LO frequency
                lo = channel.local_oscillator
                frequency = channel.lo_frequency
                power = channel.lo_power
                if lo.is_connected:
                    lo.setup(frequency=frequency, power=channel.power)
                else:
                    log.warn(f"There is no connection to {lo}. Frequencies were not set.")
                # update LO frequency in the QM config
                for qubit, mode in channel.qubits:
                    self.config["elements"][f"{mode}{qubit}"]["mixInputs"]["lo_frequency"] = frequency
                    self.config["mixers"][f"mixer_{mode}{qubit}"][0]["lo_frequency"] = frequency

    def start(self):
        # TODO: Start the OPX flux offsets?
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.stop()
            self.manager.close_all_quantum_machines()

    def disconnect(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.disconnect()
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

    def register_element(self, qubit):
        if qubit.drive:
            self.config["elements"][f"drive{qubit}"] = {
                "mixInputs": {
                    "I": qubit.drive.ports[0],
                    "Q": qubit.drive.ports[1],
                    "lo_frequency": 0,
                    "mixer": f"mixer_drive{qubit}",
                },
                "intermediate_frequency": 0,
                "operations": {},
            }
            drive_g = qubit.characterization.mixer_drive_g
            drive_phi = qubit.characterization.mixer_drive_phi
            self.config["mixers"][f"mixer_drive{qubit}"] = [
                {
                    "intermediate_frequency": 0,
                    "lo_frequency": 0,
                    "correction": self.iq_imbalance(drive_g, drive_phi),
                }
            ]
        if qubit.readout:
            self.config["elements"][f"readout{qubit}"] = {
                "mixInputs": {
                    "I": qubit.readout.ports[0],
                    "Q": qubit.readout.ports[1],
                    "lo_frequency": 0,
                    "mixer": f"mixer_readout{qubit}",
                },
                "intermediate_frequency": 0,
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
                    "intermediate_frequency": 0,
                    "lo_frequency": 0,
                    "correction": self.iq_imbalance(readout_g, readout_phi),
                }
            ]
        if qubit.flux:
            self.config["elements"][f"flux{qubit}"] = {
                "singleInput": {
                    "port": qubit.flux.ports[0],
                },
                "intermediate_frequency": 0,
                "operations": {},
            }

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
        elements = self.config["elements"]
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
                # register drive pulse in elements
                elements[f"drive{qubit}"]["operations"][pulse.serial] = pulse.serial
                if_frequency = pulse.frequency - qubit.drive.lo_frequency
                elements[f"drive{qubit}"]["intermediate_frequency"] = if_frequency
                mixers[f"mixer_drive{qubit}"][0]["intermediate_frequency"] = if_frequency

            elif pulse.type.name == "FLUX":
                serial = self.register_waveform(pulse)
                pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": serial,
                    },
                }
                # register flux pulse in elements
                elements[f"flux{qubit}"]["operations"][pulse.serial] = pulse.serial
                elements[f"flux{qubit}"]["intermediate_frequency"] = pulse.frequency

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
                # register readout pulse in elements
                elements[f"readout{qubit}"]["operations"][pulse.serial] = pulse.serial
                if_frequency = pulse.frequency - qubit.readout.lo_frequency
                elements[f"readout{qubit}"]["intermediate_frequency"] = if_frequency
                mixers[f"mixer_readout{qubit}"][0]["intermediate_frequency"] = if_frequency

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

        return f"{pulse.type.name.lower()}{str(qubit)}"

    def execute_program(self, program):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.

        Returns:
            TODO
        """
        machine = self.manager.open_qm(self.config)
        return machine.execute(program)

    def sweep_frequency(self, frequencies, sequence, nshots=1024):
        from qm.qua import (
            align,
            declare,
            declare_stream,
            dual_demod,
            fixed,
            for_,
            measure,
            play,
            program,
            save,
            stream_processing,
            update_frequency,
            wait,
        )
        from qualang_tools.loops import from_array

        # TODO: Read qubit from sweeper
        qubit = sequence.pulses[0].qubit
        nfreq = len(frequencies)
        if_frequencies = frequencies - self.qubits[qubit].readout.lo_frequency
        if_frequencies = if_frequencies.astype(int)

        targets = [self.register_pulse(pulse) for pulse in sequence]
        # play pulses using QUA
        clock = {target: 0 for target in targets}
        with program() as experiment:
            n = declare(int)
            freq = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < nshots, n + 1):
                with for_(*from_array(freq, if_frequencies)):
                    update_frequency(f"readout{qubit}", freq)
                    for pulse, target in zip(sequence, targets):
                        wait_time = pulse.start - clock[target]
                        if wait_time > 0:
                            wait(wait_time // 4, target)
                        clock[target] += pulse.duration
                        if pulse.type.name == "READOUT":
                            # align("qubit", "resonator")
                            measure(
                                pulse.serial,
                                target,
                                None,
                                dual_demod.full("cos", "out1", "sin", "out2", I),
                                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                            )
                        else:
                            play(pulse.serial, target)
                    # Wait for the resonator to cooldown
                    wait(2000 // 4, f"readout{qubit}")
                    # Save data to the stream processing
                    save(I, I_st)
                    save(Q, Q_st)
            align()

            with stream_processing():
                # I_st.average().save("I")
                # Q_st.average().save("Q")
                # n_st.buffer().save_all("n")
                I_st.buffer(nfreq).average().save("I")
                Q_st.buffer(nfreq).average().save("Q")

        return self.execute_program(experiment)

    def play(self, qubits, sequence, nshots=1024):
        """Plays an arbitrary pulse sequence on the instruments.

        Args:
            qubits (list): List of :class:`qibo.platforms.utils.Qubit` objects representing
                the qubits the instruments are acting on.
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to play.
            nshots (int): Number of (hardware) repetitions for the execution.

        Returns:
            TODO
        """
        from qm.qua import (
            declare,
            declare_stream,
            dual_demod,
            fixed,
            for_,
            measure,
            play,
            program,
            save,
            stream_processing,
            wait,
        )

        # TODO: Fix phases
        # TODO: Handle pulses that run on the same element simultaneously (multiplex?)
        # register pulses in Quantum Machines config
        targets = [self.register_pulse(qubits[pulse.qubit], pulse) for pulse in sequence]
        # play pulses using QUA
        clock = {target: 0 for target in targets}
        with program() as experiment:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < nshots, n + 1):
                for pulse, target in zip(sequence, targets):
                    wait_time = pulse.start - clock[target]
                    if wait_time > 0:
                        wait(wait_time // 4, target)
                    clock[target] += pulse.duration
                    if pulse.type.name == "READOUT":
                        # align("qubit", "resonator")
                        measure(
                            pulse.serial,
                            target,
                            None,
                            dual_demod.full("cos", "out1", "sin", "out2", I),
                            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                        )
                    else:
                        play(pulse.serial, target)
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)

            with stream_processing():
                # I_st.average().save("I")
                # Q_st.average().save("Q")
                # n_st.buffer().save_all("n")
                I_st.buffer(nshots).save("I")
                Q_st.buffer(nshots).save("Q")

        return self.execute_program(experiment)
