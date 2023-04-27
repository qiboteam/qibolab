""" Qblox instruments driver.

Supports the following Instruments:
    Cluster
    Cluster QRM-RF
    Cluster QCM-RF
    Cluster QCM
Compatible with qblox-instruments driver 0.9.0 (28/2/2023).
It supports:
    - multiplexed readout of up to 6 qubits
    - hardware modulation, demodulation and classification
    - software modulation, with support for arbitrary pulses
    - software demodulation
    - binned acquisition (max bins 131_072)
    - real-time sweepers of frequency, gain, offset, pulse start and pulse duration
    - max iq pulse length 8_192ns
    - waveforms cache, uses additional free sequencers if the memory of one sequencer (16384) is exhausted
    - intrument parameters cache

The operation of multiple clusters symultaneously is not supported yet.
https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/
"""

import json
from enum import Enum, auto

import numpy as np
from qblox_instruments.qcodes_drivers.cluster import Cluster as QbloxCluster
from qblox_instruments.qcodes_drivers.qcm_qrm import QcmQrm as QbloxQrmQcm
from qblox_instruments.qcodes_drivers.sequencer import Sequencer as QbloxSequencer

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.instruments.qblox_q1asm import (
    Block,
    Program,
    Register,
    convert_frequency,
    convert_gain,
    convert_offset,
    convert_phase,
    loop_block,
    wait_block,
)
from qibolab.pulses import Pulse, PulseSequence, PulseShape, PulseType, Waveform
from qibolab.sweeper import Parameter, Sweeper


class QbloxSweeperType(Enum):
    """An enumeration for the different types of sweepers supported by qblox.

    frequency: sweeps pulse frequency by adjusting the sequencer `nco_freq` with q1asm command `set_freq`.
    gain: sweeps sequencer gain by adjusting the sequencer `gain_awg_path0` and `gain_awg_path1` with q1asm command
        `set_awg_gain`. Since the gain is a parameter between -1 and 1 that multiplies the samples of the waveforms
        before they are fed to the DACs, it can be used to sweep the pulse amplitude.
    offset: sweeps sequencer offset by adjusting the sequencer `offset_awg_path0` and `offset_awg_path1` with q1asm
        command `set_awg_offs`
    start: sweeps pulse start.
    duration: sweeps pulse duration.
    """

    frequency = auto()
    gain = auto()
    offset = auto()
    start = auto()
    duration = auto()

    number = auto()  # internal
    phase = auto()  # not implemented yet
    time = auto()  # not implemented yet


class QbloxSweeper:
    """A custom sweeper object with the data and functionality required by qblox.

    It is responsible for generating the q1asm code required to execute sweeps in a sequencer. The object can be
        initialised either with:
            a :class:`qibolab.sweepers.Sweeper` using the :func:`qibolab.instruments.qblox.QbloxSweeper.from_sweeper`, or
            a range of values and a sweeper type (:class:`qibolab.instruments.qblox.QbloxSweeperType`)

    Attributes:
        type (:class:`qibolab.instruments.qblox.QbloxSweeperType`): the type of sweeper
        name (str): a name given for the sweep that is later used within the q1asm code to identify the loops.
        register (:class:`qibolab.instruments.qblox_q1asm.Register`): the main Register (q1asm variable) used in the loop.
        aux_register (:class:`qibolab.instruments.qblox_q1asm.Register`): an auxialiry Register requried in duration
            sweeps.
        update_parameters (Bool): a flag to instruct the sweeper to update the paramters or not depending on whether
            a parameter of the sequencer needs to be swept or not.

    Methods:
        block(inner_block: :class:`qibolab.instruments.qblox_q1asm.Block): generates the block of q1asm code that implements
            the sweep.
    """

    FREQUENCY_LIMIT = 500e6

    @classmethod
    def from_sweeper(
        cls, program: Program, sweeper: Sweeper, add_to: float = 0, multiply_to: float = 1, name: str = ""
    ):
        """Creates an instance form a :class:`qibolab.sweepers.Sweeper` object.

        Args:
            program (:class:`qibolab.instruments.qblox_q1asm.Program`): a program object representing the q1asm program of a
                sequencer.
            sweeper (:class:`qibolab.sweepers.Sweeper`): the original qibolab sweeper.
                associated with the sweep. If no name is provided it uses the sweeper type as name.
            add_to (float): a value to be added to each value of the range of values defined in `sweeper.values`,
                `rel_values`.
            multiply_to (float): a value to be multiplied by each value of the range of values defined in `sweeper.values`,
                `rel_values`.
            name (str): a name given for the sweep that is later used within the q1asm code to identify the loops.
        """
        type_c = {
            Parameter.frequency: QbloxSweeperType.frequency,
            Parameter.gain: QbloxSweeperType.gain,
            Parameter.amplitude: QbloxSweeperType.gain,
            Parameter.bias: QbloxSweeperType.offset,
            Parameter.start: QbloxSweeperType.start,
            Parameter.duration: QbloxSweeperType.duration,
            Parameter.relative_phase: QbloxSweeperType.phase,
        }
        if sweeper.parameter in type_c:
            type = type_c[sweeper.parameter]
            rel_values = sweeper.values
        else:
            raise ValueError(f"Sweeper parameter {sweeper.parameter} is not supported by qblox driver yet.")
        return cls(program=program, rel_values=rel_values, type=type, add_to=add_to, multiply_to=multiply_to, name=name)

    def __init__(
        self,
        program: Program,
        rel_values: list,
        type: QbloxSweeperType = QbloxSweeperType.number,
        add_to: float = 0,
        multiply_to: float = 1,
        name: str = "",
    ):
        """Creates an instance from a range of values and a sweeper type (:class:`qibolab.instruments.qblox.QbloxSweeperType`).

        Args:
            program (:class:`qibolab.instruments.qblox_q1asm.Program`): a program object representing the q1asm program
                of a sequencer.
            rel_values (list): a list of values to iterate over. Currently qblox only supports a list of equaly spaced
                values, like those created with `np.arange(start, stop, step)`. These values are considered relative
                values. They will later be added to the `add_to` parameter and multiplied to the `multiply_to`
                parameter.
            type (:class:`qibolab.instruments.qblox.QbloxSweeperType`): the type of sweeper.
            add_to (float): a value to be added to each value of the range of values defined in `sweeper.values` or
                `rel_values`.
            multiply_to (float): a value to be multiplied by each value of the range of values defined in
            `sweeper.values` or `rel_values`.
            name (str): a name given for the sweep that is later used within the q1asm code to identify the loops.
        """

        self.type: QbloxSweeperType = type
        self.name: str = None
        self.register: Register = None
        self.aux_register: Register = None
        self.update_parameters: bool = False

        # Number of iterations in the loop
        self._n: int = None

        # Absolute values
        self._abs_start = None
        self._abs_step = None
        self._abs_stop = None
        self._abs_values: np.ndarray = None

        # Converted values (converted to q1asm values)
        self._con_start: int = None
        self._con_step: int = None
        self._con_stop: int = None
        self._con_values: np.ndarray = None

        # Validate input parameters
        if not len(rel_values) > 1:
            raise ValueError("values must contain at least 2 elements.")
        elif rel_values[1] == rel_values[0]:
            raise ValueError("values must contain at different elements.")

        self._n = len(rel_values) - 1
        rel_start = rel_values[0]
        rel_step = rel_values[1] - rel_values[0]

        if name != "":
            self.name = name
        else:
            self.name = self.type.name

        # create the registers (variables) to be used in the loop
        self.register: Register = Register(program, self.name)
        if type == QbloxSweeperType.duration:
            self.aux_register: Register = Register(program, self.name + "_aux")

        # Calculate absolute values
        self._abs_start = (rel_start + add_to) * multiply_to
        self._abs_step = rel_step * multiply_to
        self._abs_stop = self._abs_start + self._abs_step * (self._n)
        self._abs_values = np.arange(self._abs_start, self._abs_stop, self._abs_step)

        # Verify that all values are within acceptable ranges
        check_values = {
            QbloxSweeperType.frequency: (
                lambda v: all((-self.FREQUENCY_LIMIT <= x and x <= self.FREQUENCY_LIMIT) for x in v)
            ),
            QbloxSweeperType.gain: (lambda v: all((-1 <= x and x <= 1) for x in v)),
            QbloxSweeperType.offset: (lambda v: all((-1.25 * np.sqrt(2) <= x and x <= 1.25 * np.sqrt(2)) for x in v)),
            QbloxSweeperType.phase: (lambda v: True),
            QbloxSweeperType.start: (lambda v: all((4 <= x and x < 2**16) for x in v)),
            QbloxSweeperType.duration: (lambda v: all((0 <= x and x < 2**16) for x in v)),
            QbloxSweeperType.number: (lambda v: all((-(2**16) < x and x < 2**16) for x in v)),
        }

        if not check_values[type](np.append(self._abs_values, [self._abs_stop])):
            raise ValueError(f"Sweeper {self.name} values are not within the allowed range")

        # Convert absolute values to q1asm values
        convert = {
            QbloxSweeperType.frequency: convert_frequency,
            QbloxSweeperType.gain: convert_gain,
            QbloxSweeperType.offset: convert_offset,
            QbloxSweeperType.phase: convert_phase,
            QbloxSweeperType.start: (lambda x: int(x) % 2**16),
            QbloxSweeperType.duration: (lambda x: int(x) % 2**16),
            QbloxSweeperType.number: (lambda x: int(x) % 2**32),
        }

        self._con_start = convert[type](self._abs_start)
        self._con_step = convert[type](self._abs_step)
        self._con_stop = (self._con_start + self._con_step * (self._n) + 1) % 2**32
        self._con_values = np.array([(self._con_start + self._con_step * m) % 2**32 for m in range(self._n + 1)])

        if not (
            isinstance(self._con_start, int) and isinstance(self._con_stop, int) and isinstance(self._con_step, int)
        ):
            raise ValueError("start, stop and step must be int")

    def block(self, inner_block: Block):
        """Generates the block of q1asm code that implements the sweep.

        Args:
            inner_block (:class:`qibolab.instruments.qblox_q1asm.Block): the block of q1asm code to be repeated within
                the loop.

        """
        # Initialisation
        header_block = Block()
        header_block.append(
            f"move {self._con_start}, {self.register}",
            comment=f"{self.register.name} loop, start: {round(self._abs_start, 6):_}",
        )
        header_block.append("nop")
        header_block.append(f"loop_{self.register}:")

        # Parameter update
        if self.update_parameters:
            update_parameter_block = Block()
            update_time = 1000
            if self.type == QbloxSweeperType.frequency:
                update_parameter_block.append(f"set_freq {self.register}")  # move to pulse
                update_parameter_block.append(f"upd_param {update_time}")
            if self.type == QbloxSweeperType.gain:
                update_parameter_block.append(f"set_awg_gain {self.register}, {self.register}")  # move to pulse
                update_parameter_block.append(f"upd_param {update_time}")
            if self.type == QbloxSweeperType.offset:
                update_parameter_block.append(f"set_awg_offs {self.register}, {self.register}")
                update_parameter_block.append(f"upd_param {update_time}")

            if self.type == QbloxSweeperType.start:
                pass
            if self.type == QbloxSweeperType.duration:
                update_parameter_block.append(f"add {self.register}, 1, {self.aux_register}")
            if self.type == QbloxSweeperType.time:
                pass
            if self.type == QbloxSweeperType.number:
                pass
            header_block += update_parameter_block
        header_block.append_spacer()

        # Main code
        body_block = Block()
        body_block.indentation = 1
        body_block += inner_block

        # Loop instructions
        footer_block = Block()
        footer_block.append_spacer()

        footer_block.append(
            f"add {self.register}, {self._con_step}, {self.register}",
            comment=f"{self.register.name} loop, step: {round(self._abs_step, 6):_}",
        )
        footer_block.append("nop")

        if self._abs_step > 0:  # increasing
            if (self._abs_start < 0 and self._abs_stop < 0) or (
                self._abs_stop > 0 and self._abs_start >= 0
            ):  # no crossing
                footer_block.append(
                    f"jlt {self.register}, {self._con_stop}, @loop_{self.register}",
                    comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                )
            elif self._abs_start < 0 and self._abs_stop >= 0:  # crossing
                footer_block.append(
                    f"jge {self.register}, {2**31}, @loop_{self.register}",
                )
                footer_block.append(
                    f"jlt {self.register}, {self._con_stop}, @loop_{self.register}",
                    comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                )
            else:
                raise ValueError(
                    f"incorrect values for abs_start: {self._abs_start}, abs_stop: {self._abs_stop}, abs_step: {self._abs_step}"
                )
        elif self._abs_step < 0:  # decreasing
            if (self._abs_start < 0 and self._abs_stop < 0) or (
                self._abs_stop >= 0 and self._abs_start > 0
            ):  # no crossing
                footer_block.append(
                    f"jge {self.register}, {self._con_stop + 1}, @loop_{self.register}",
                    comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                )
            elif self._abs_start >= 0 and self._abs_stop < 0:  # crossing
                if self._con_stop + 1 != 2**32:
                    footer_block.append(
                        f"jlt {self.register}, {2**31}, @loop_{self.register}",
                    )
                    footer_block.append(
                        f"jge {self.register}, {self._con_stop + 1}, @loop_{self.register}",
                        comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                    )
                else:
                    footer_block.append(
                        f"jlt {self.register}, {2**31}, @loop_{self.register}",
                        comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                    )
            else:
                raise ValueError(
                    f"incorrect values for abs_start: {self._abs_start}, abs_stop: {self._abs_stop}, abs_step: {self._abs_step}"
                )

        return header_block + body_block + footer_block


class WaveformsBuffer:
    """A class to represent a buffer that holds the unique waveforms used by a sequencer.

    Attributes:
        unique_waveforms (list): A list of unique Waveform objects.
        available_memory (int): The amount of memory available expressed in numbers of samples.
    """

    SIZE: int = 16383

    class NotEnoughMemory(Exception):
        """An error raised when there is not enough memory left to add more waveforms."""

        pass

    def __init__(self):
        """Initialises the buffer with an empty list of unique waveforms."""
        self.unique_waveforms: list = []  # Waveform
        self.available_memory: int = WaveformsBuffer.SIZE

    def add_waveforms(self, pulse: Pulse, hardware_mod_en: bool):
        """Adds a pair of i and q waveforms to the list of unique waveforms.

        Waveforms are added to the list if they were not there before.
        Each of the waveforms (i and q) is processed individually.

        Args:
            waveform_i (Waveform): A Waveform object containing the samples of the real component of the pulse wave.
            waveform_q (Waveform): A Waveform object containing the samples of the imaginary component of the pulse wave.

        Raises:
            NotEnoughMemory: If the memory needed to store the waveforms in more than the memory avalible.
        """
        if hardware_mod_en:
            waveform_i, waveform_q = pulse.envelope_waveforms
        else:
            waveform_i, waveform_q = pulse.modulated_waveforms

        pulse.waveform_i = waveform_i
        pulse.waveform_q = waveform_q

        if waveform_i not in self.unique_waveforms or waveform_q not in self.unique_waveforms:
            memory_needed = 0
            if not waveform_i in self.unique_waveforms:
                memory_needed += len(waveform_i)
            if not waveform_q in self.unique_waveforms:
                memory_needed += len(waveform_q)

            if self.available_memory >= memory_needed:
                if not waveform_i in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_i)
                if not waveform_q in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_q)
                self.available_memory -= memory_needed
            else:
                raise WaveformsBuffer.NotEnoughMemory

    def bake_pulse_waveforms(
        self, pulse: Pulse, values: list(), hardware_mod_en: bool
    ):  # bake_pulse_waveforms(self, pulse: Pulse, values: list(int), hardware_mod_en: bool):
        """Generates and stores a set of i and q waveforms required for a pulse duration sweep.

        These waveforms are generated and stored in a predefined order so that they can later be retrieved within the
        sweeper q1asm code. It bakes pulses from as short as 1ns, padding them at the end with 0s if required so that
        their length is a multiple of 4ns. It also supports the modulation of the pulse both in hardware (default)
            or software.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): The pulse to be swept.
            values (list(int)): The list of values to sweep the pulse duration with.
            hardware_mod_en (bool): If set to True the pulses are assumed to be modulated in hardware and their
                evelope waveforms are uploaded; if False, the software modulated waveforms are uploaded.

        Returns:
            idx_range (numpy.ndarray): An array with the indices of the set of pulses. For each pulse duration in
                `values` the i component is saved in the next avalable index, followed by the q component. For flux
                pulses, since both i and q components are equal, they are only saved once.

        Raises:
            NotEnoughMemory: If the memory needed to store the waveforms in more than the memory avalible.
        """
        # In order to generate waveforms for each duration value, the pulse duration will need to be modified.
        # To avoid any conflict, make a copy of the pulse first
        p = pulse.copy()

        first_idx = len(self.unique_waveforms)
        if pulse.type == PulseType.FLUX:
            idx_range = np.arange(first_idx, first_idx + len(values), 1)

            for duration in values:
                p.duration = duration
                if hardware_mod_en:
                    waveform = p.envelope_waveform_i
                else:
                    waveform = p.modulated_waveform_i

                padded_duration = int(np.ceil(duration / 4)) * 4
                memory_needed = padded_duration
                padding = np.zeros(padded_duration - duration)
                waveform.data = np.append(waveform.data, padding)

                if self.available_memory >= memory_needed:
                    self.unique_waveforms.append(waveform)
                    self.available_memory -= memory_needed
                else:
                    raise WaveformsBuffer.NotEnoughMemory
        else:
            idx_range = np.arange(first_idx, first_idx + len(values) * 2, 2)

            for duration in values:
                p.duration = duration
                if hardware_mod_en:
                    waveform_i, waveform_q = p.envelope_waveforms
                else:
                    waveform_i, waveform_q = p.modulated_waveforms

                padded_duration = int(np.ceil(duration / 4)) * 4
                memory_needed = padded_duration * 2
                padding = np.zeros(padded_duration - duration)
                waveform_i.data = np.append(waveform_i.data, padding)
                waveform_q.data = np.append(waveform_q.data, padding)

                if self.available_memory >= memory_needed:
                    self.unique_waveforms.append(waveform_i)
                    self.unique_waveforms.append(waveform_q)
                    self.available_memory -= memory_needed
                else:
                    raise WaveformsBuffer.NotEnoughMemory

        return idx_range


class Sequencer:
    """A class to extend the functionality of qblox_instruments Sequencer.

    A sequencer is a hardware component synthesised in the instrument FPGA, responsible for fetching waveforms from
    memory, pre-processing them, sending them to the DACs, and processing the acquisitions from the ADCs (QRM modules).
    https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html

    This class extends the sequencer functionality by holding additional data required when
    processing a pulse sequence:
        the sequencer number,
        the sequence of pulses to be played,
        a buffer of unique waveforms, and
        the four components of the sequence file:
            waveforms dictionary
            acquisition dictionary
            weights dictionary
            program

    Attributes:
        device (QbloxSequencer): A reference to the underlying `qblox_instruments.qcodes_drivers.sequencer.Sequencer`
            object. It can be used to access other features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html
        number (int): An integer between 0 and 5 that identifies the number of the sequencer.
        pulses (PulseSequence): The sequence of pulses to be played by the sequencer.
        waveforms_buffer (WaveformsBuffer): A buffer of unique waveforms to be played by the sequencer.
        waveforms (dict): A dictionary containing the waveforms to be played by the sequencer in qblox format.
        acquisitions (dict): A dictionary containing the list of acquisitions to be made by the sequencer in qblox
            format.
        weights (dict): A dictionary containing the list of weights to be used by the sequencer when demodulating
            and integrating the response, in qblox format.
        program (str): The pseudo assembly (q1asm) program to be executed by the sequencer.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html#instructions

        qubit (str): The id of the qubit associated with the sequencer, if there is only one.
    """

    def __init__(self, number: int):
        """Initialises the sequencer.

        All class attributes are defined and initialised.
        """

        self.device: QbloxSequencer = None
        self.number: int = number
        self.pulses: PulseSequence = PulseSequence()
        self.waveforms_buffer: WaveformsBuffer = WaveformsBuffer()
        self.waveforms: dict = {}
        self.acquisitions: dict = {}
        self.weights: dict = {}
        self.program: Program = Program()
        self.qubit = None  # self.qubit: int | str = None


class Cluster(AbstractInstrument):
    """A class to extend the functionality of qblox_instruments Cluster.

    The class exposes the attribute `reference_clock_source` to enable the selection of an internal or external clock
    source.

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        device (QbloxCluster): A reference to the underlying `qblox_instruments.qcodes_drivers.cluster.Cluster` object.
            It can be used to access other features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/cluster.html
        reference_clock_source (str): ('internal', 'external') Instructs the cluster to use the internal clock source
            or an external source.
    """

    #
    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    property_wrapper.__doc__ = """A lambda function used to create properties that wrap around device parameters and
    caches their value using `_set_device_parameter()`.
    """

    def __init__(self, name: str, address: str):
        """Initialises the instrument storing its name and address."""
        super().__init__(name, address)
        # self.reference_clock_source: str
        self.device: QbloxCluster = None

        self._device_parameters = {}

    def connect(self):
        """Connects to the instrument.

        If the connection is successful, it saves a reference to the underlying object in the attribute `device`.
        The instrument is reset on each connection.
        """
        global cluster
        if not self.is_connected:
            for attempt in range(3):
                try:
                    QbloxCluster.close_all()
                    self.device = QbloxCluster(self.name, self.address)
                    self.device.reset()
                    cluster = self.device
                    self.is_connected = True
                    # DEBUG: Print Cluster Status
                    # print(self.device.get_system_status())
                    setattr(
                        self.__class__,
                        "reference_clock_source",
                        self.property_wrapper("reference_source"),
                    )  # FIXME: this will not work when using multiple instances, replace with explicit properties
                    break
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            pass

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument if its value changed from the last value stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters (list): A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument with the settings saved in the runcard.

        Args:
            **kwargs: A dictionary with the settings:
                reference_clock_source
        """
        if self.is_connected:
            self.reference_clock_source = kwargs["reference_clock_source"]
        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
        self.device.close()
        global cluster
        cluster = None
        # TODO: set modules is_connected to False


cluster: QbloxCluster = None
# TODO: In order to support the symultaneous operation of multiple clusters, replace the global variable
#  with a collection of clusters implemented as a static property of the class Cluster.


class ClusterQRM_RF(AbstractInstrument):
    """Qblox Cluster Qubit Readout Module RF driver.

    Qubit Readout Module RF (QRM-RF) is an instrument that integrates an arbitratry wave generator, a digitizer,
    a local oscillator and a mixer. It has one output and one input ports. Each port has a path0 and path1 for the
    i(in-phase) and q(quadrature) components of the RF signal. The sampling rate of its ADC/DAC is 1 GSPS.
    https://www.qblox.com/cluster

    The class aims to simplify the configuration of the instrument, exposing only those parameters most frequencly
    used and hiding other more complex settings.

    A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QRM_QCM` object is provided via the
    attribute `device`, allowing the advanced user to gain access to the features that are not exposed directly
    by the class.

    In order to accelerate the execution, the instrument settings are cached, so that the communication with the
    instrument only happens when the parameters change. This caching is done with the method
    `_set_device_parameter(target, *parameters, value)`.

        ports:
            o1:                                             # output port settings
                channel                     : L3-25a
                attenuation                 : 30                # (dB) 0 to 60, must be multiple of 2
                lo_enabled                  : true
                lo_frequency                : 6_000_000_000     # (Hz) from 2e9 to 18e9
                gain                        : 1                 # for path0 and path1 -1.0<=v<=1.0
            i1:                                             # input port settings
                channel                     : L2-5a
                acquisition_hold_off        : 130               # minimum 4ns
                acquisition_duration        : 1800

        classification_parameters:
            0: # qubit id
                rotation_angle              : 0                 # in degrees 0.0<=v<=360.0
                threshold                   : 0                 # in V
            1:
                rotation_angle              : 194.272
                threshold                   : 0.011197
            2:
                rotation_angle              : 104.002
                threshold                   : 0.012745

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        name (str): A unique name given to the instrument.
        address (str): IP_address:module_number; the IP address of the cluster and
            the module number.
        device (QbloxQrmQcm): A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm` object.
            It can be used to access other features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html

        ports = A dictionary giving access to the input and output ports objects.
            ports['o1']: Output port
            ports['i1']: Input port

            ports['o1'].channel (int | str): the id of the refrigerator channel the output port o1 is connected to.
            ports['o1'].attenuation (int): (mapped to qrm.out0_att) Controls the attenuation applied to the output
                port. It must be a multiple of 2.
            ports['o1'].lo_enabled (bool): (mapped to qrm.out0_in0_lo_en) Enables or disables the local oscillator.
            ports['o1'].lo_frequency (int): (mapped to qrm.out0_in0_lo_freq) Sets the frequency of the local oscillator.
            ports['o1'].gain (float): (mapped to qrm.sequencers[0].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o1'].hardware_mod_en (bool): (mapped to qrm.sequencers[0].mod_en_awg) Enables pulse modulation
                in hardware. When set to False, pulse modulation is done in software, at the host computer, and the
                modulated pulse waveform is uploaded to the instrument. When set to True, the envelope of the pulse
                is uploaded to the instrument and it is modulated in real time by the FPGA of the instrument, using
                the sequencer nco (numerically controlled oscillator).
            ports['o1'].nco_freq (int): (mapped to qrm.sequencers[0].nco_freq). Mapped, but not configurable from
                the runcard.
            ports['o1'].nco_phase_offs = (mapped to qrm.sequencers[0].nco_phase_offs). Mapped, but not configurable
                from the runcard.

            ports['i1'].channel (int | str): the id of the refrigerator channel the input port o1 is connected to.
            ports['i1'].acquisition_hold_off (int): Delay between the moment the readout pulse starts to be played and
                the start of the acquisition, in ns. It must be > 0 and multiple of 4.
            ports['i1'].acquisition_duration (int): (mapped to qrm.sequencers[0].integration_length_acq) Duration
            of the pulse acquisition, in ns. It must be > 0 and multiple of 4.
            ports['i1'].hardware_demod_en (bool): (mapped to qrm.sequencers[0].demod_en_acq) Enables demodulation
                and integration of the acquired pulses in hardware. When set to False, the filtration, demodulation
                and integration of the acquired pulses is done at the host computer. When set to True, the
                demodulation, integration and discretization of the pulse is done in real time at the FPGA of the
                instrument.

        classification_parameters (dict): A dictionary containing the paramters needed classify the state of each qubit.
            from a single shot measurement:
                qubit_id (dict): the id of the qubit
                    rotation_angle (float): 0   # in degrees 0.0<=v<=360.0. The angle of the rotation applied at the
                        origin of the i q plane, that put the centroids of the state |0> and state |1> in a horizontal line.
                        The rotation puts the centroid of state |1> to the right side of centroid of state |0>.
                    threshold (float): 0        # in V. The the real component of the point along the horizontal line
                        connecting both state centroids (after being rotated), that maximises the fidelity of the
                        classification.

        channels (list): A list of the channels to which the instrument is connected.

        Sequencer 0 is used always for acquisitions and it is the first sequencer used to synthesise pulses.
        Sequencer 1 to 6 are used as needed to sinthesise simultaneous pulses on the same channel (required in
        multipled readout) or when the memory of the default sequencers rans out.

    """

    DEFAULT_SEQUENCERS: dict = {"o1": 0, "i1": 0}
    SAMPLING_RATE: int = 1e9  # 1 GSPS
    FREQUENCY_LIMIT = 500e6

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    property_wrapper.__doc__ = """A lambda function used to create properties that wrap around the device parameters
    and caches their value using `_set_device_parameter()`.
    """
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device.sequencers[sequencer], *parameter, value=x),
    )
    sequencer_property_wrapper.__doc__ = """A lambda function used to create properties that wrap around the device
    sequencer parameters and caches their value using `_set_device_parameter()`.
    """

    def __init__(self, name: str, address: str):
        """Initialises the instance.

        All class attributes are defined and initialised.
        """

        super().__init__(name, address)
        self.device: QbloxQrmQcm = None
        self.ports: dict = {}
        self.classification_parameters: dict = {}
        self.channels: list = []

        self._cluster: QbloxCluster = None
        self._input_ports_keys = ["i1"]
        self._output_ports_keys = ["o1"]
        self._sequencers: dict[Sequencer] = {"o1": []}
        self._port_channel_map: dict = {}
        self._channel_port_map: dict = {}
        self._device_parameters = {}
        self._device_num_output_ports = 1
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []
        self._execution_time: float = 0

    def connect(self):
        """Connects to the instrument using the instrument settings in the runcard.

        Once connected, it creates port classes with properties mapped to various instrument
        parameters, and initialises the the underlying device parameters.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                # save a reference to the underlying object
                self.device: QbloxQrmQcm = cluster.modules[int(self.address.split(":")[1]) - 1]

                # create a class for each port with attributes mapped to the instrument parameters
                self.ports["o1"] = type(
                    f"port_o1",
                    (),
                    {
                        "channel": None,
                        "attenuation": self.property_wrapper("out0_att"),
                        "lo_enabled": self.property_wrapper("out0_in0_lo_en"),
                        "lo_frequency": self.property_wrapper("out0_in0_lo_freq"),
                        "gain": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "gain_awg_path0", "gain_awg_path1"
                        ),
                        "hardware_mod_en": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "mod_en_awg"),
                        "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "nco_freq"),
                        "nco_phase_offs": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "nco_phase_offs"
                        ),
                    },
                )()
                self.ports["i1"] = type(
                    f"port_i1",
                    (),
                    {
                        "channel": None,
                        "acquisition_hold_off": 0,
                        "acquisition_duration": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "integration_length_acq"
                        ),
                        "hardware_demod_en": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["i1"], "demod_en_acq"
                        ),
                    },
                )()

                # save reference to cluster
                self._cluster = cluster
                self.is_connected = True

                # once connected, initialise the parameters of the device to the default values
                self._set_device_parameter(self.device, "in0_att", value=0)
                self._set_device_parameter(
                    self.device, "out0_offset_path0", "out0_offset_path1", value=0
                )  # Default after reboot = 7.625
                self._set_device_parameter(
                    self.device, "scope_acq_avg_mode_en_path0", "scope_acq_avg_mode_en_path1", value=True
                )
                self._set_device_parameter(
                    self.device, "scope_acq_sequencer_select", value=self.DEFAULT_SEQUENCERS["o1"]
                )
                self._set_device_parameter(
                    self.device, "scope_acq_trigger_level_path0", "scope_acq_trigger_level_path1", value=0
                )
                self._set_device_parameter(
                    self.device, "scope_acq_trigger_mode_path0", "scope_acq_trigger_mode_path1", value="sequencer"
                )

                # initialise the parameters of the default sequencer to the default values,
                # the rest of the sequencers are not configured here, but will be configured
                # with the same parameters as the default in process_pulse_sequence()
                target = self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]]

                self._set_device_parameter(target, "channel_map_path0_out0_en", "channel_map_path1_out1_en", value=True)
                self._set_device_parameter(target, "cont_mode_en_awg_path0", "cont_mode_en_awg_path1", value=False)
                self._set_device_parameter(
                    target, "cont_mode_waveform_idx_awg_path0", "cont_mode_waveform_idx_awg_path1", value=0
                )
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0
                self._set_device_parameter(target, "mixer_corr_gain_ratio", value=1)
                self._set_device_parameter(target, "mixer_corr_phase_offset_degree", value=0)
                self._set_device_parameter(target, "offset_awg_path0", value=0)
                self._set_device_parameter(target, "offset_awg_path1", value=0)
                self._set_device_parameter(target, "sync_en", value=False)  # Default after reboot = False
                self._set_device_parameter(target, "upsample_rate_awg_path0", "upsample_rate_awg_path1", value=0)

                # on initialisation, disconnect all other sequencers from the ports
                self._device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(1, self._device_num_sequencers):
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out0_en",
                        "channel_map_path1_out1_en",
                        value=False,
                    )  # Default after reboot = True

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters (list): A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument with the settings of the runcard.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['ports']['o1']['channel'] (int | str): the id of the refrigerator channel the output port o1 is
                    connected to.
                kwargs['ports']['o1']['attenuation'] (int): [0 to 60 dBm, in multiples of 2] attenuation at the output.
                kwargs['ports']['o1']['lo_enabled'] (bool): enable or disable local oscillator for up-conversion.
                kwargs['ports']['o1']['lo_frequency'] (int): [2_000_000_000 to 18_000_000_000 Hz] local oscillator
                    frequency.
                kwargs['ports']['o1']['gain'] (float): [0.0 - 1.0 unitless] gain applied prior to up-conversion. Qblox
                    recommends to keep `pulse_amplitude * gain` below 0.3 to ensure the mixers are working in their
                    linear regime, if necessary, lowering the attenuation applied at the output.
                kwargs['ports']['o1']['hardware_mod_en'] (bool): enables Hardware Modulation. In this mode, pulses are
                    modulated to the intermediate frequency using the numerically controlled oscillator within the
                    fpga. It only requires the upload of the pulse envelope waveform.
                kwargs['ports']['i1']['channel'] (int | str): the id of the refrigerator channel the input port i1 is
                    connected to.
                kwargs['ports']['i1']['hardware_demod_en'] (bool): enables Hardware Demodulation. In this mode, the
                    sequencers of the fpga demodulate, integrate and classify the results for every shot. Once
                    integrated, the i and q values and the result of the classification requires much less memory,
                    so they can be stored for every shot in separate `bins` and retrieved later. Hardware Demodulation
                    also allows making multiple readouts on the same qubit at different points in the circuit, which is
                    not possible with Software Demodulation.
                kwargs['acquisition_hold_off'] (int): [0 to 16834 ns, in multiples of 4] the time between the moment
                    the start of the readout pulse begins to be played, and the start of the acquisition. This is used
                    to account for the time of flight of the pulses from the output port to the input port.
                kwargs['acquisition_duration'] (int): [0 to 8192 ns] the duration of the acquisition. It is limited by
                    the amount of memory available in the fpga to store i q samples.
                kwargs['classification_parameters'][qubit_id][rotation_angle] (float): [0.0 to 359.999 deg] the angle
                    to rotate the results so that the projection on the real axis renders the maximum fidelity.
                kwargs['classification_parameters'][qubit_id][threshold] (float): [-1.0 to 1.0 Volt] the voltage to be
                    used as threshold to classify the state of each shot.

        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            self.ports["o1"].channel = kwargs["ports"]["o1"]["channel"]
            self._port_channel_map["o1"] = self.ports["o1"].channel
            self.ports["o1"].attenuation = kwargs["ports"]["o1"]["attenuation"]
            self.ports["o1"].lo_enabled = kwargs["ports"]["o1"]["lo_enabled"]
            self.ports["o1"].lo_frequency = kwargs["ports"]["o1"]["lo_frequency"]
            self.ports["o1"].gain = kwargs["ports"]["o1"]["gain"]

            if "hardware_mod_en" in kwargs["ports"]["o1"]:
                self.ports["o1"].hardware_mod_en = kwargs["ports"]["o1"]["hardware_mod_en"]
            else:
                self.ports["o1"].hardware_mod_en = True

            self.ports["o1"].nco_freq = 0
            self.ports["o1"].nco_phase_offs = 0

            self.ports["i1"].channel = kwargs["ports"]["i1"]["channel"]
            self._port_channel_map["i1"] = self.ports["i1"].channel
            if "hardware_demod_en" in kwargs["ports"]["i1"]:
                self.ports["i1"].hardware_demod_en = kwargs["ports"]["i1"]["hardware_demod_en"]
            else:
                self.ports["i1"].hardware_demod_en = True

            self.ports["i1"].acquisition_hold_off = kwargs["ports"]["i1"]["acquisition_hold_off"]
            self.ports["i1"].acquisition_duration = kwargs["ports"]["i1"]["acquisition_duration"]

            self._channel_port_map = {v: k for k, v in self._port_channel_map.items()}
            self.channels = list(self._channel_port_map.keys())
            if "classification_parameters" in kwargs:
                self.classification_parameters = kwargs["classification_parameters"]

            self._last_pulsequence_hash = 0

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def _get_next_sequencer(self, port: str, frequency: int, qubit: None):
        """Retrieves and configures the next avaliable sequencer.

        The parameters of the new sequencer are copied from those of the default sequencer, except for the intermediate
        frequency and classification parameters.

        Args:
            port (str):
            frequency (int):
            qubit (str|int):
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        # select a new sequencer and configure it as required
        next_sequencer_number = self._free_sequencers_numbers.pop(0)
        if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                # exclude read-only parameter `sequence` and others that have wrong default values (qblox bug)
                if not parameter in ["sequence", "thresholded_acq_marker_address", "thresholded_acq_trigger_address"]:
                    value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                    if value:
                        target = self.device.sequencers[next_sequencer_number]
                        self._set_device_parameter(target, parameter, value=value)

        # if hardware demodulation is enabled, configure nco_frequency and classification parameters
        if self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en:
            self._set_device_parameter(
                self.device.sequencers[next_sequencer_number],
                "nco_freq",
                value=frequency,
                # It assumes all pulses in non_overlapping_pulses set have the same frequency.
                # Non-overlapping pulses of different frequencies on the same qubit channel, with hardware_demod_en
                # would lead to wrong results.
                # TODO: Throw error in that event or implement  non_overlapping_same_frequency_pulses
            )
        if self.ports["i1"].hardware_demod_en and qubit in self.classification_parameters:
            self._set_device_parameter(
                self.device.sequencers[next_sequencer_number],
                "thresholded_acq_rotation",
                value=self.classification_parameters[qubit]["rotation_angle"],
            )
            self._set_device_parameter(
                self.device.sequencers[next_sequencer_number],
                "thresholded_acq_threshold",
                value=self.classification_parameters[qubit]["threshold"] * self.ports["i1"].acquisition_duration,
            )
        # create sequencer wrapper
        sequencer = Sequencer(next_sequencer_number)
        sequencer.qubit = qubit
        return sequencer

    def get_if(self, pulse: Pulse):
        """Returns the intermediate frequency needed to synthesise a pulse based on the port lo frequency."""

        _rf = pulse.frequency
        _lo = self.ports[self._channel_port_map[pulse.channel]].lo_frequency
        _if = _rf - _lo
        if abs(_if) > self.FREQUENCY_LIMIT:
            raise RuntimeError(
                f"""
            Pulse frequency {_rf} cannot be synthesised with current lo frequency {_lo}.
            The intermediate frequency {_if} would exceed the maximum frequency of {self.FREQUENCY_LIMIT}
            """
            )
        return _if

    def process_pulse_sequence(
        self,
        instrument_pulses: PulseSequence,
        navgs: int,
        nshots: int,
        repetition_duration: int,
        sweepers: list() = [],  # sweepers: list(Sweeper) = []
    ):
        """Processes a list of pulses, generating the waveforms and sequence program required by
        the instrument to synthesise them.

        The output of the process is a list of sequencers used for each port, configured with the information
        required to play the sequence.
        The following features are supported:
            - multiplexed readout of up to 6 qubits
            - overlapping pulses
            - hardware modulation, demodulation and classification
            - software modulation, with support for arbitrary pulses
            - software demodulation
            - binned acquisition (max bins 131_072)
            - real-time sweepers of frequency, gain, offset, pulse start and pulse duration
            - max iq pulse length 8_192ns
            - waveforms cache, uses additional free sequencers if the memory of one sequencer (16384) is exhausted
            - intrument parameters cache

        Args:
            instrument_pulses (PulseSequence): A collection of Pulse objects to be played by the instrument.
            navgs (int): The number of times the sequence of pulses should be executed averaging the results.
            nshots (int): The number of times the sequence of pulses should be executed.
            repetition_duration (int): The total duration of the pulse sequence execution plus the reset/relaxation time.
            sweepers (list(Sweeper)): A list of Sweeper objects to be implemented.
        """

        # calculate the number of bins
        num_bins = nshots
        for sweeper in sweepers:
            num_bins *= len(sweeper.values)

        # estimate the execution time
        self._execution_time = navgs * num_bins * ((repetition_duration + 1000 * len(sweepers)) * 1e-9)

        port = "o1"
        # initialise the list of free sequencer numbers to include the default for each port {'o1': 0}
        self._free_sequencers_numbers = [self.DEFAULT_SEQUENCERS[port]] + [1, 2, 3, 4, 5]

        # split the collection of instruments pulses by ports
        port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self._port_channel_map[port])

        # initialise the list of sequencers required by the port
        self._sequencers[port] = []

        if not port_pulses.is_empty:
            # split the collection of port pulses in non overlapping pulses
            non_overlapping_pulses: PulseSequence
            for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                # each set of not overlapping pulses will be played by a separate sequencer
                # check sequencer availability
                if len(self._free_sequencers_numbers) == 0:
                    raise Exception(
                        f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                    )
                # get next sequencer
                sequencer = self._get_next_sequencer(
                    port=port,
                    frequency=self.get_if(non_overlapping_pulses[0]),
                    qubit=non_overlapping_pulses[0].qubit,
                )
                # add the sequencer to the list of sequencers required by the port
                self._sequencers[port].append(sequencer)

                # make a temporary copy of the pulses to be processed
                pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                while not pulses_to_be_processed.is_empty:
                    pulse: Pulse = pulses_to_be_processed[0]
                    # attempt to save the waveforms to the sequencer waveforms buffer
                    try:
                        sequencer.waveforms_buffer.add_waveforms(pulse, self.ports[port].hardware_mod_en)
                        sequencer.pulses.add(pulse)
                        pulses_to_be_processed.remove(pulse)

                    # if there is not enough memory in the current sequencer, use another one
                    except WaveformsBuffer.NotEnoughMemory:
                        if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                            raise NotImplementedError(
                                f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported."
                            )
                        if len(self._free_sequencers_numbers) == 0:
                            raise Exception(
                                f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                            )
                        # get next sequencer
                        sequencer = self._get_next_sequencer(
                            port=port,
                            frequency=self.get_if(non_overlapping_pulses[0]),
                            qubit=non_overlapping_pulses[0].qubit,
                        )
                        # add the sequencer to the list of sequencers required by the port
                        self._sequencers[port].append(sequencer)

        # update the lists of used and unused sequencers that will be needed later on
        self._used_sequencers_numbers = []
        for port in self._output_ports_keys:
            for sequencer in self._sequencers[port]:
                self._used_sequencers_numbers.append(sequencer.number)
        self._unused_sequencers_numbers = []
        for n in range(self._device_num_sequencers):
            if not n in self._used_sequencers_numbers:
                self._unused_sequencers_numbers.append(n)

        # generate and store the Waveforms dictionary, the Acquisitions dictionary, the Weights and the Program
        for port in self._output_ports_keys:
            for sequencer in self._sequencers[port]:
                pulses = sequencer.pulses
                program = sequencer.program

                ## pre-process sweepers ##

                # attach a sweeper attribute to the pulse so that it is easily accesible by the code that generates
                # the pseudo-assembly program
                pulse = None
                for pulse in pulses:
                    pulse.sweeper = None

                # define whether to sweep relative values or absolute values depending on the type of sweep
                # TODO: incorporate add_to and multiply_to to qibolab.Sweeper so that the user can decide
                # until then:
                #   frequency - relative values added to the pulse frequency
                #   amplitude, gain, offset, start, duration - absolute values
                for sweeper in sweepers:
                    reference_value = 0
                    if sweeper.parameter == Parameter.frequency:
                        if sequencer.pulses:
                            reference_value = self.get_if(sequencer.pulses[0])

                    # if sweeper.parameter == Parameter.amplitude:
                    #     reference_value = self.ports[port].gain
                    # if sweeper.parameter == Parameter.bias:
                    #     reference_value = self.ports[port].offset

                    # create QbloxSweepers and attach them to qibolab sweeper
                    if sweeper.parameter == Parameter.duration and pulse in sweeper.pulses:
                        # for duration sweepers bake waveforms
                        idx_range = sequencer.waveforms_buffer.bake_pulse_waveforms(
                            pulse, sweeper.values, self.ports[port].hardware_mod_en
                        )
                        sweeper.qs = QbloxSweeper(program=program, type=QbloxSweeperType.duration, rel_values=idx_range)
                    else:
                        sweeper.qs = QbloxSweeper.from_sweeper(program=program, sweeper=sweeper, add_to=reference_value)

                    # FIXME: for qubit sweepers (Parameter.bias, Parameter.attenuation, Parameter.gain), the qubit
                    # information alone is not enough to determine whether this sequencer should actively participate
                    # in this sweep or not. One may want to change a parameter that is not associated with a pulse,
                    # for example port gain, but knowing the qubit is not enough, as both the drive and readout ports
                    # associated with one qubit, have a gain parameter.
                    # until this is resolved, and since bias/offset is only implemented on QCMs, this instrument will
                    # never take an active role in those sweeps:

                    # if sweeper.qubits and sequencer.qubit in sweeper.qubits:
                    #     sweeper.qs.update_parameters = True

                    # finally attach QbloxSweepers to the pulses being swept
                    if sweeper.pulses:
                        for pulse in pulses:
                            if pulse in sweeper.pulses:
                                sweeper.qs.update_parameters = True
                                pulse.sweeper = sweeper.qs

                # Waveforms
                for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                    sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                # Acquisitions
                for acquisition_index, pulse in enumerate(sequencer.pulses.ro_pulses):
                    sequencer.acquisitions[pulse.serial] = {"num_bins": num_bins, "index": acquisition_index}

                # Add scope_acquisition to default sequencer
                if sequencer.number == self.DEFAULT_SEQUENCERS[port]:
                    sequencer.acquisitions["scope_acquisition"] = {"num_bins": 1, "index": acquisition_index + 1}

                # Program
                minimum_delay_between_instructions = 4

                # Active reset is not fully tested yet
                active_reset = False
                active_reset_address = 1
                active_reset_pulse_idx_I = 1
                active_reset_pulse_idx_Q = 1

                sequence_total_duration = pulses.finish  # the minimum delay between instructions is 4ns
                time_between_repetitions = repetition_duration - sequence_total_duration
                assert time_between_repetitions > minimum_delay_between_instructions
                # relaxation_time needs to be greater than acquisition_hold_off

                nshots_register = Register(program, "nshots")
                navgs_register = Register(program, "navgs")
                bin_n = Register(program, "bin_n")

                header_block = Block("setup")
                if active_reset:
                    header_block.append(
                        f"set_latch_en {active_reset_address}, 4", f"monitor triggers on address {active_reset_address}"
                    )

                body_block = Block()

                body_block.append(f"wait_sync {minimum_delay_between_instructions}")
                if self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en:
                    body_block.append("reset_ph")
                    body_block.append_spacer()

                pulses_block = Block("play_and_acquire")
                # Add an initial wait instruction for the first pulse of the sequence
                initial_wait_block = wait_block(
                    wait_time=pulses[0].start, register=Register(program), force_multiples_of_4=True
                )
                pulses_block += initial_wait_block

                for n in range(pulses.count):
                    if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.start:
                        pulses_block.append(f"wait {pulses[n].sweeper.register}")

                    if self.ports["o1"].hardware_mod_en:
                        # # Set frequency
                        # _if = self.get_if(pulses[n])
                        # pulses_block.append(f"set_freq {convert_frequency(_if)}", f"set intermediate frequency to {_if} Hz")

                        # Set phase
                        pulses_block.append(
                            f"set_ph {convert_phase(pulses[n].relative_phase)}",
                            comment=f"set relative phase {pulses[n].relative_phase} rads",
                        )

                    if pulses[n].type == PulseType.READOUT:
                        delay_after_play = self.ports["i1"].acquisition_hold_off

                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_acquire = (
                                pulses[n + 1].start - pulses[n].start - self.ports["i1"].acquisition_hold_off
                            )
                        else:
                            delay_after_acquire = sequence_total_duration - pulses[n].start
                            time_between_repetitions = (
                                repetition_duration - sequence_total_duration - self.ports["i1"].acquisition_hold_off
                            )
                            assert time_between_repetitions > 0

                        if delay_after_acquire < minimum_delay_between_instructions:
                            raise Exception(
                                f"The minimum delay after starting acquisition is {minimum_delay_between_instructions}ns."
                            )

                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.duration:
                            RI = pulses[n].sweeper.register
                            if pulses[n].type == PulseType.FLUX:
                                RQ = pulses[n].sweeper.register
                            else:
                                RQ = pulses[n].sweeper.aux_register

                            pulses_block.append(
                                f"play  {RI},{RQ},{delay_after_play}",  # FIXME delay_after_play won't work as the duration increases
                                comment=f"play pulse {pulses[n]} sweeping its duration",
                            )
                        else:
                            # Prepare play instruction: play wave_i_index, wave_q_index, delay_next_instruction
                            pulses_block.append(
                                f"play  {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}",
                                comment=f"play waveforms {pulses[n]}",
                            )

                        # Prepare acquire instruction: acquire acquisition_index, bin_index, delay_next_instruction
                        if active_reset:
                            pulses_block.append(f"acquire {pulses.ro_pulses.index(pulses[n])},{bin_n},4")
                            pulses_block.append(f"latch_rst {delay_after_acquire + 300 - 4}")
                        else:
                            pulses_block.append(
                                f"acquire {pulses.ro_pulses.index(pulses[n])},{bin_n},{delay_after_acquire}"
                            )

                    else:
                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[n + 1].start - pulses[n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[n].start
                        if delay_after_play < minimum_delay_between_instructions:
                            raise Exception(
                                f"The minimum delay between pulses is {minimum_delay_between_instructions}ns."
                            )

                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.duration:
                            RI = pulses[n].sweeper.register
                            if pulses[n].type == PulseType.FLUX:
                                RQ = pulses[n].sweeper.register
                            else:
                                RQ = pulses[n].sweeper.aux_register

                            pulses_block.append(
                                f"play  {RI},{RQ},{delay_after_play}",  # FIXME delay_after_play won't work as the duration increases
                                comment=f"play pulse {pulses[n]} sweeping its duration",
                            )
                        else:
                            # Prepare play instruction: play wave_i_index, wave_q_index, delay_next_instruction
                            pulses_block.append(
                                f"play  {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}",
                                comment=f"play waveforms {pulses[n]}",
                            )

                body_block += pulses_block
                body_block.append_spacer()

                if active_reset:
                    final_reset_block = Block()
                    final_reset_block.append(f"set_cond 1, {active_reset_address}, 0, 4", comment="active reset")
                    final_reset_block.append(f"play {active_reset_pulse_idx_I}, {active_reset_pulse_idx_Q}, 4", level=1)
                    final_reset_block.append(f"set_cond 0, {active_reset_address}, 0, 4")
                else:
                    final_reset_block = wait_block(
                        wait_time=time_between_repetitions, register=Register(program), force_multiples_of_4=False
                    )
                final_reset_block.append_spacer()
                final_reset_block.append(f"add {bin_n}, 1, {bin_n}", "increase bin counter")

                body_block += final_reset_block

                footer_block = Block("cleaup")
                footer_block.append(f"stop")

                for sweeper in sweepers:
                    body_block = sweeper.qs.block(inner_block=body_block)

                nshots_block: Block = loop_block(
                    start=0, stop=nshots, step=1, register=nshots_register, block=body_block
                )
                nshots_block.prepend(f"move 0, {bin_n}", "reset bin counter")
                nshots_block.append_spacer()

                navgs_block = loop_block(start=0, stop=navgs, step=1, register=navgs_register, block=nshots_block)
                program.add_blocks(header_block, navgs_block, footer_block)

                sequencer.program = repr(program)

    def upload(self):
        """Uploads waveforms and programs of all sequencers and arms them in preparation for execution.

        This method should be called after `process_pulse_sequence()`.
        It configures certain parameters of the instrument based on the needs of resources determined
        while processing the pulse sequence.
        """
        # Setup
        for sequencer_number in self._used_sequencers_numbers:
            target = self.device.sequencers[sequencer_number]
            self._set_device_parameter(target, "sync_en", value=True)
            self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
            self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0

        for sequencer_number in self._unused_sequencers_numbers:
            target = self.device.sequencers[sequencer_number]
            self._set_device_parameter(target, "sync_en", value=False)
            self._set_device_parameter(target, "marker_ovr_en", value=False)  # Default after reboot = False
            self._set_device_parameter(target, "marker_ovr_value", value=0)  # Default after reboot = 0
            if sequencer_number >= 1:  # Never disconnect default sequencers
                self._set_device_parameter(target, "channel_map_path0_out0_en", value=False)
                self._set_device_parameter(target, "channel_map_path1_out1_en", value=False)

        # Upload waveforms and program
        qblox_dict = {}
        sequencer: Sequencer
        for port in self._output_ports_keys:
            for sequencer in self._sequencers[port]:
                # Add sequence program and waveforms to single dictionary
                qblox_dict[sequencer] = {
                    "waveforms": sequencer.waveforms,
                    "weights": sequencer.weights,
                    "acquisitions": sequencer.acquisitions,
                    "program": sequencer.program,
                }

                # Upload dictionary to the device sequencers
                self.device.sequencers[sequencer.number].sequence(qblox_dict[sequencer])

                # DEBUG: QRM RF Save sequence to file
                filename = f"Z_{self.name}_sequencer{sequencer.number}_sequence.json"
                with open(filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)
                    file.write(sequencer.program)

        # Clear acquisition memory and arm sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.sequencers[sequencer_number].delete_acquisition_data(all=True)
            self.device.arm_sequencer(sequencer_number)

        # DEBUG: QRM RF Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

        # DEBUG: QRM RF Save Readable Snapshot
        # filename = f"Z_{self.name}_snapshot.json"
        # with open(filename, "w", encoding="utf-8") as file:
        #     print_readable_snapshot(self.device, file, update=True)

    def play_sequence(self):
        """Plays the sequence of pulses.

        Starts the sequencers needed to play the sequence of pulses."""

        # Start used sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.start_sequencer(sequencer_number)

    def acquire(self):
        """Retrieves the readout results.

        Returns:
            The results returned vary depending on whether demodulation is performed in software or hardware:
            - Software Demodulation:
                Every readout pulse triggers an acquisition, where the 16384 i and q samples of the waveform
                acquired by the ADC are saved into a dedicated memory within the FPGA. This is what qblox calls
                *scoped acquisition*. The results of multiple shots are averaged in this memory, and cannot be
                retrieved independently. The resulting waveforms averages (i and q) are then demodulated and
                integrated in software (and finally divided by the number of samples).
                Since Software Demodulation relies on the data of the scoped acquisition and that data is the
                average of all acquisitions, **only one readout pulse per qubit is supported**, so that
                the averages all correspond to reading the same quantum state.
                Multiple symultaneous readout pulses on different qubits are supported.
                The results returned are:
                    acquisition_results["averaged_raw"] (dict): a dictionary containing tuples with the averages of
                        the i and q waveforms for every readout pulse:
                        ([i samples], [q samples])
                        The data for a specific reaout pulse can be obtained either with:
                            `acquisition_results["averaged_raw"][ro_pulse.serial]`
                            `acquisition_results["averaged_raw"][ro_pulse.qubit]`

                    acquisition_results["averaged_demodulated_integrated"] (dict): a dictionary containing tuples
                    with the results of demodulating and integrating (averaging over time) the average of the
                    waveforms for every pulse:
                        `(amplitude[V], phase[rad], i[V], q[V])`
                    The data for a specific readout pulse can be obtained either with:
                        `acquisition_results["averaged_demodulated_integrated"][ro_pulse.serial]`
                        `acquisition_results["averaged_demodulated_integrated"][ro_pulse.qubit]`
                    Or directly with:
                        `acquisition_results[ro_pulse.serial]`
                        `acquisition_results[ro_pulse.qubit]`

            - Hardware Demodulation:
                With hardware demodulation activated, the FPGA can demodulate, integrate (average over time), and classify
                each shot individually, saving the results on separate bins. The raw data of each acquisition continues to
                be averaged as with software modulation, so there is no way to access the raw data of each shot (unless
                executed one shot at a time). The FPGA uses fixed point arithmetic for the demodulation and integration;
                if the power level of the signal at the input port is low (the minimum resolution of the ADC is 240uV)
                rounding precission errors can accumulate and render wrong results. It is advisable to have a power level
                at least higher than 5mV.

                The results returned are:
                    acquisition_results["demodulated_integrated_averaged"] (dict): a dictionary containing tuples
                    with the results of demodulating and integrating (averaging over time) each shot waveform and then
                    averaging of the many shots:
                        `(amplitude[V], phase[rad], i[V], q[V])`
                    acquisition_results["demodulated_integrated_binned"] (dict): a dictionary containing tuples of lists
                    with the results of demodulating and integrating every shot waveform:
                        `([amplitudes[V]], [phases[rad]], [is[V]], [qs[V]])`
                    acquisition_results["demodulated_integrated_classified_binned"] (dict): a dictionary containing lists
                    with the results of demodulating, integrating and classifying every shot:
                        `([states[0 or 1]])`
                    acquisition_results["probability"] (dict): a dictionary containing the frequency of state 1 measurements:
                        total number of shots classified as 1 / number of shots

                If the number of readout pulses per qubit is only one, then the following is also provided:
                    acquisition_results["averaged_raw"] (dict): a dictionary containing tuples with the averages of
                        the i and q waveforms for every readout pulse:
                        ([i samples], [q samples])
                    acquisition_results["averaged_demodulated_integrated"] (dict): a dictionary containing tuples
                    with the results of demodulating and integrating (averaging over time) the average of the
                    waveforms for every pulse:
                        `(amplitude[V], phase[rad], i[V], q[V])`

                    The data within each of the above dictionaries, for a specific readout pulse or for the last readout
                    pulse of a qubit can be retrieved either with:
                        `acquisition_results[dictionary_name][ro_pulse.serial]`
                        `acquisition_results[dictionary_name][ro_pulse.qubit]`

                    And acquisition_results["averaged_demodulated_integrated"] directly with:
                        `acquisition_results[ro_pulse.serial]`
                        `acquisition_results[ro_pulse.qubit]`

        """

        # wait until all sequencers stop
        time_out = int(self._execution_time) + 60
        import time

        from qibo.config import log

        t = time.time()

        for sequencer_number in self._used_sequencers_numbers:
            while True:
                try:
                    state = self.device.get_sequencer_state(sequencer_number)
                except:
                    pass
                else:
                    if state.status == "STOPPED":
                        # log.info(f"{self.device.sequencers[sequencer_number].name} state: {state}")
                        # TODO: check flags for errors
                        break
                    elif time.time() - t > time_out:
                        log.info(f"Timeout - {self.device.sequencers[sequencer_number].name} state: {state}")
                        self.device.stop_sequencer(sequencer_number)
                        break
                time.sleep(1)

        acquisition_results = {}

        # Qblox qrm modules only have one memory for scope acquisition.
        # Only one sequencer can save data to that memory.
        # Several acquisitions at different points in the circuit will result in the undesired averaging
        # of the results.
        # Scope Acquisition should only be used with one acquisition per module.
        # Several readout pulses are supported for as long as they take place symultaneously.
        # Scope Acquisition data should be ignored with more than one acquisition or with Hardware Demodulation.

        # Software Demodulation requires the data from Scope Acquisition, therefore Software Demodulation only works
        # with one acquisition per module.

        # The data is retrieved by storing it first in one of the acquisitions of one of the sequencers.
        # Any could be used, but we always use 'scope_acquisition' acquisition of the default sequencer to store it.

        # Store scope acquisition data on 'scope_acquisition' acquisition of the default sequencer
        for port in self._output_ports_keys:
            for sequencer in self._sequencers[port]:
                if sequencer.number == self.DEFAULT_SEQUENCERS[port]:
                    self.device.store_scope_acquisition(sequencer.number, "scope_acquisition")
                    scope_acquisition_raw_results = self.device.get_acquisitions(sequencer.number)["scope_acquisition"]

        acquisition_results["demodulated_integrated_averaged"] = {}
        acquisition_results["averaged_raw"] = {}
        acquisition_results["averaged_demodulated_integrated"] = {}
        acquisition_results["demodulated_integrated_binned"] = {}
        acquisition_results["demodulated_integrated_classified_binned"] = {}
        acquisition_results["probability"] = {}
        data = {}
        for port in self._output_ports_keys:
            for sequencer in self._sequencers[port]:
                if not self.ports["i1"].hardware_demod_en:  # Software Demodulation
                    if len(sequencer.pulses.ro_pulses) == 1:
                        pulse = sequencer.pulses.ro_pulses[0]

                        acquisition_results["averaged_raw"][pulse.serial] = (
                            scope_acquisition_raw_results["acquisition"]["scope"]["path0"]["data"][
                                0 : self.ports["i1"].acquisition_duration
                            ],
                            scope_acquisition_raw_results["acquisition"]["scope"]["path1"]["data"][
                                0 : self.ports["i1"].acquisition_duration
                            ],
                        )
                        acquisition_results["averaged_raw"][pulse.qubit] = acquisition_results["averaged_raw"][
                            pulse.serial
                        ]

                        i, q = self._process_acquisition_results(scope_acquisition_raw_results, pulse, demodulate=True)
                        acquisition_results["averaged_demodulated_integrated"][pulse.serial] = (
                            np.sqrt(i**2 + q**2),
                            np.arctan2(q, i),
                            i,
                            q,
                        )
                        acquisition_results["averaged_demodulated_integrated"][pulse.qubit] = acquisition_results[
                            "averaged_demodulated_integrated"
                        ][pulse.serial]

                        # Default Results = Averaged Demodulated Integrated
                        acquisition_results[pulse.serial] = acquisition_results["averaged_demodulated_integrated"][
                            pulse.serial
                        ]
                        acquisition_results[pulse.qubit] = acquisition_results["averaged_demodulated_integrated"][
                            pulse.qubit
                        ]

                        # ignores the data available in acquisition results and returns only i and q voltages
                        # TODO: to be updated once the functionality of ExecutionResults is extended
                        data[pulse.serial] = (i, q)

                    else:
                        raise Exception(
                            """Software Demodulation only supports one acquisition per channel.
                        Multiple readout pulses are supported as long as they are symultaneous (requiring one acquisition)"""
                        )

                else:  # Hardware Demodulation
                    binned_raw_results = self.device.get_acquisitions(sequencer.number)
                    for pulse in sequencer.pulses.ro_pulses:
                        acquisition_name = pulse.serial
                        i, q = self._process_acquisition_results(
                            binned_raw_results[acquisition_name], pulse, demodulate=False
                        )
                        acquisition_results["demodulated_integrated_averaged"][pulse.serial] = (
                            np.sqrt(i**2 + q**2),
                            np.arctan2(q, i),
                            i,
                            q,
                        )
                        acquisition_results["demodulated_integrated_averaged"][pulse.qubit] = acquisition_results[
                            "demodulated_integrated_averaged"
                        ][pulse.serial]

                        # Default Results = Demodulated Integrated Averaged
                        acquisition_results[pulse.serial] = acquisition_results["demodulated_integrated_averaged"][
                            pulse.serial
                        ]
                        acquisition_results[pulse.qubit] = acquisition_results["demodulated_integrated_averaged"][
                            pulse.qubit
                        ]

                        # Save individual shots
                        shots_i = (
                            np.array(
                                binned_raw_results[acquisition_name]["acquisition"]["bins"]["integration"]["path0"]
                            )
                            / self.ports["i1"].acquisition_duration
                        )
                        shots_q = (
                            np.array(
                                binned_raw_results[acquisition_name]["acquisition"]["bins"]["integration"]["path1"]
                            )
                            / self.ports["i1"].acquisition_duration
                        )

                        acquisition_results["demodulated_integrated_binned"][pulse.serial] = (
                            np.sqrt(shots_i**2 + shots_q**2),
                            np.arctan2(shots_q, shots_i),
                            shots_i,
                            shots_q,
                        )
                        acquisition_results["demodulated_integrated_binned"][pulse.qubit] = acquisition_results[
                            "demodulated_integrated_binned"
                        ][pulse.serial]

                        acquisition_results["demodulated_integrated_classified_binned"][
                            pulse.serial
                        ] = binned_raw_results[acquisition_name]["acquisition"]["bins"]["threshold"]
                        acquisition_results["demodulated_integrated_classified_binned"][
                            pulse.qubit
                        ] = acquisition_results["demodulated_integrated_classified_binned"][pulse.serial]

                        acquisition_results["probability"][pulse.serial] = np.mean(
                            acquisition_results["demodulated_integrated_classified_binned"][pulse.serial]
                        )
                        acquisition_results["probability"][pulse.qubit] = acquisition_results["probability"][
                            pulse.serial
                        ]

                        # Provide Scope Data for verification (assuming memory reseet is being done)
                        if len(sequencer.pulses.ro_pulses) == 1:
                            pulse = sequencer.pulses.ro_pulses[0]

                            acquisition_results["averaged_raw"][pulse.serial] = (
                                scope_acquisition_raw_results["acquisition"]["scope"]["path0"]["data"][
                                    0 : self.ports["i1"].acquisition_duration
                                ],
                                scope_acquisition_raw_results["acquisition"]["scope"]["path1"]["data"][
                                    0 : self.ports["i1"].acquisition_duration
                                ],
                            )
                            acquisition_results["averaged_raw"][pulse.qubit] = acquisition_results["averaged_raw"][
                                pulse.serial
                            ]

                            i, q = self._process_acquisition_results(
                                scope_acquisition_raw_results, pulse, demodulate=True
                            )

                            acquisition_results["averaged_demodulated_integrated"][pulse.serial] = (
                                np.sqrt(i**2 + q**2),
                                np.arctan2(q, i),
                                i,
                                q,
                            )
                            acquisition_results["averaged_demodulated_integrated"][pulse.qubit] = acquisition_results[
                                "averaged_demodulated_integrated"
                            ][pulse.serial]

                        # ignores the data available in acquisition results and returns only i and q voltages
                        # TODO: to be updated once the functionality of ExecutionResults is extended
                        data[acquisition_name] = (
                            shots_i,
                            shots_q,
                            np.array(acquisition_results["demodulated_integrated_classified_binned"][acquisition_name]),
                        )

                    # DEBUG: QRM RF Plot Incomming Pulses
                    # import qibolab.instruments.debug.incomming_pulse_plotting as pp
                    # pp.plot(raw_results)
                    # DEBUG: QRM RF Plot Acquisition_results
                    # from qibolab.debug.debug import plot_acquisition_results
                    # plot_acquisition_results(acquisition_results, pulse, savefig_filename="acquisition_results.png")
        return data

    def _process_acquisition_results(self, acquisition_results, readout_pulse: Pulse, demodulate=True):
        """Processes the results of the acquisition.

        If hardware demodulation is disabled, it demodulates and integrates the acquired pulse. If enabled,
        if processes the results as required by qblox (calculating the average by dividing the integrated results by
        the number of smaples acquired).
        """
        if demodulate:
            acquisition_frequency = self.get_if(readout_pulse)

            # DOWN Conversion
            n0 = 0
            n1 = self.ports["i1"].acquisition_duration
            input_vec_I = np.array(acquisition_results["acquisition"]["scope"]["path0"]["data"][n0:n1])
            input_vec_Q = np.array(acquisition_results["acquisition"]["scope"]["path1"]["data"][n0:n1])
            input_vec_I -= np.mean(input_vec_I)  # qblox does not remove the offsets in hardware
            input_vec_Q -= np.mean(input_vec_Q)

            modulated_i = input_vec_I
            modulated_q = input_vec_Q

            num_samples = modulated_i.shape[0]
            time = np.arange(num_samples) / PulseShape.SAMPLING_RATE

            cosalpha = np.cos(2 * np.pi * acquisition_frequency * time)
            sinalpha = np.sin(2 * np.pi * acquisition_frequency * time)
            demod_matrix = np.sqrt(2) * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time, modulated_i, modulated_q):
                result.append(demod_matrix[:, :, it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = np.mean(demodulated_signal, axis=0)

            # import matplotlib.pyplot as plt
            # plt.plot(input_vec_I[:400])
            # plt.plot(list(map(list, zip(*demodulated_signal)))[0][:400])
            # plt.show()
        else:
            i = np.mean(
                np.array(acquisition_results["acquisition"]["bins"]["integration"]["path0"])
                / self.ports["i1"].acquisition_duration
            )
            q = np.mean(
                np.array(acquisition_results["acquisition"]["bins"]["integration"]["path1"])
                / self.ports["i1"].acquisition_duration
            )
            integrated_signal = i, q
        return integrated_signal

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Stops all sequencers"""
        try:
            self.device.stop_sequencer()
        except:
            pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        self._cluster = None
        self.is_connected = False


class ClusterQCM_RF(AbstractInstrument):
    """Qblox Cluster Qubit Control Module RF driver.

    Qubit Control Module RF (QCM-RF) is an instrument that integrates an arbitratry
    wave generator, with a local oscillator and a mixer. It has two output ports
    Each port has a path0 and path1 for the i(in-phase) and q(quadrature) components
    of the RF signal. The sampling rate of its ADC/DAC is 1 GSPS.
    https://www.qblox.com/cluster

    The class aims to simplify the configuration of the instrument, exposing only
    those parameters most frequencly used and hiding other more complex components.

    A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QRM_QCM`
    object is provided via the attribute `device`, allowing the advanced user to gain
    access to the features that are not exposed directly by the class.

    In order to accelerate the execution, the instrument settings are cached, so that
    the communication with the instrument only happens when the parameters change.
    This caching is done with the method `_set_device_parameter(target, *parameters, value)`.

        ports:
            o1:                     # output port settings
                channel                      : L3-11
                attenuation                  : 24               # (dB) 0 to 60, must be multiple of 2
                lo_enabled                   : true
                lo_frequency                 : 4_042_590_000    # (Hz) from 2e9 to 18e9
                gain                         : 0.17             # for path0 and path1 -1.0<=v<=1.0
            o2:
                channel                      : L3-12
                attenuation                  : 24               # (dB) 0 to 60, must be multiple of 2
                lo_enabled                   : true
                lo_frequency                 : 5_091_155_529    # (Hz) from 2e9 to 18e9
                gain                         : 0.28             # for path0 and path1 -1.0<=v<=1.0

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        name (str): A unique name given to the instrument.
        address (str): IP_address:module_number (the IP address of the cluster and
            the module number)
        device (QbloxQrmQcm): A reference to the underlying
            `qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm` object. It can be used to access other
            features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html

        ports = A dictionary giving access to the output ports objects.
            ports['o1']
            ports['o2']

            ports['o1'].channel (int | str): the id of the refrigerator channel the port is connected to.
            ports['o1'].attenuation (int): (mapped to qrm.out0_att) Controls the attenuation applied to
                the output port. Must be a multiple of 2
            ports['o1'].lo_enabled (bool): (mapped to qrm.out0_in0_lo_en) Enables or disables the
                local oscillator.
            ports['o1'].lo_frequency (int): (mapped to qrm.out0_in0_lo_freq) Sets the frequency of the
                local oscillator.
            ports['o1'].gain (float): (mapped to qrm.sequencers[0].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o1'].hardware_mod_en (bool): (mapped to qrm.sequencers[0].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['o1'].nco_freq (int): (mapped to qrm.sequencers[0].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['o1'].nco_phase_offs = (mapped to qrm.sequencers[0].nco_phase_offs) # TODO mapped, but not configurable from the runcard

            ports['o2'].channel (int | str): the id of the refrigerator channel the port is connected to.
            ports['o2'].attenuation (int): (mapped to qrm.out1_att) Controls the attenuation applied to
                the output port. Must be a multiple of 2
            ports['o2'].lo_enabled (bool): (mapped to qrm.out1_lo_en) Enables or disables the
                local oscillator.
            ports['o2'].lo_frequency (int): (mapped to qrm.out1_lo_freq) Sets the frequency of the
                local oscillator.
            ports['o2'].gain (float): (mapped to qrm.sequencers[1].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o2'].hardware_mod_en (bool): (mapped to qrm.sequencers[1].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['o2'].nco_freq (int): (mapped to qrm.sequencers[1].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['o2'].nco_phase_offs = (mapped to qrm.sequencers[1].nco_phase_offs) # TODO mapped, but not configurable from the runcard

        channels (list): A list of the channels to which the instrument is connected.

        Sequencer 0 is always the first sequencer used to synthesise pulses on port o1.
        Sequencer 1 is always the first sequencer used to synthesise pulses on port o2.
        Sequencer 2 to 6 are used as needed to sinthesise simultaneous pulses on the same channel
        or when the memory of the default sequencers rans out.

    """

    DEFAULT_SEQUENCERS = {"o1": 0, "o2": 1}
    SAMPLING_RATE: int = 1e9  # 1 GSPS
    FREQUENCY_LIMIT = 500e6

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    property_wrapper.__doc__ = """A lambda function used to create properties that wrap around the device parameters and
    caches their value using `_set_device_parameter()`.
    """
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device.sequencers[sequencer], *parameter, value=x),
    )
    sequencer_property_wrapper.__doc__ = """A lambda function used to create properties that wrap around the device sequencer
    parameters and caches their value using `_set_device_parameter()`.
    """

    def __init__(self, name: str, address: str):
        """Initialises the instance.

        All class attributes are defined and initialised.
        """
        super().__init__(name, address)
        self.device: QbloxQrmQcm = None
        self.ports: dict = {}
        self.channels: list = []

        self._cluster: QbloxCluster = None
        self._output_ports_keys = ["o1", "o2"]
        self._sequencers: dict[Sequencer] = {"o1": [], "o2": []}
        self._port_channel_map: dict = {}
        self._channel_port_map: dict = {}
        self._last_pulsequence_hash: int = 0
        self._current_pulsesequence_hash: int
        self._device_parameters = {}
        self._device_num_output_ports = 2
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []

    def connect(self):
        """Connects to the instrument using the instrument settings in the runcard.

        Once connected, it creates port classes with properties mapped to various instrument
        parameters, and initialises the the underlying device parameters.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device = cluster.modules[int(self.address.split(":")[1]) - 1]
                for n in range(2):
                    port = "o" + str(n + 1)
                    self.ports[port] = type(
                        f"port_" + port,
                        (),
                        {
                            "channel": None,
                            "attenuation": self.property_wrapper(f"out{n}_att"),
                            "lo_enabled": self.property_wrapper(f"out{n}_lo_en"),
                            "lo_frequency": self.property_wrapper(f"out{n}_lo_freq"),
                            "gain": self.sequencer_property_wrapper(
                                self.DEFAULT_SEQUENCERS[port], "gain_awg_path0", "gain_awg_path1"
                            ),
                            "hardware_mod_en": self.sequencer_property_wrapper(
                                self.DEFAULT_SEQUENCERS[port], "mod_en_awg"
                            ),
                            "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS[port], "nco_freq"),
                            "nco_phase_offs": self.sequencer_property_wrapper(
                                self.DEFAULT_SEQUENCERS[port], "nco_phase_offs"
                            ),
                        },
                    )()

                # save reference to cluster
                self._cluster = cluster
                self.is_connected = True

                # once connected, initialise the parameters of the device to the default values
                self._set_device_parameter(
                    self.device, "out0_offset_path0", "out0_offset_path1", value=0
                )  # Default after reboot = 7.625
                self._set_device_parameter(
                    self.device, "out1_offset_path0", "out1_offset_path1", value=0
                )  # Default after reboot = 7.625

                # initialise the parameters of the default sequencers to the default values,
                # the rest of the sequencers are not configured here, but will be configured
                # with the same parameters as the default in process_pulse_sequence()
                for target in [
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                ]:
                    self._set_device_parameter(target, "cont_mode_en_awg_path0", "cont_mode_en_awg_path1", value=False)
                    self._set_device_parameter(
                        target, "cont_mode_waveform_idx_awg_path0", "cont_mode_waveform_idx_awg_path1", value=0
                    )
                    self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                    self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0
                    self._set_device_parameter(target, "mixer_corr_gain_ratio", value=1)
                    self._set_device_parameter(target, "mixer_corr_phase_offset_degree", value=0)
                    self._set_device_parameter(target, "offset_awg_path0", value=0)
                    self._set_device_parameter(target, "offset_awg_path1", value=0)
                    self._set_device_parameter(target, "sync_en", value=False)  # Default after reboot = False
                    self._set_device_parameter(target, "upsample_rate_awg_path0", "upsample_rate_awg_path1", value=0)

                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    "channel_map_path0_out0_en",
                    "channel_map_path1_out1_en",
                    value=True,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    "channel_map_path0_out2_en",
                    "channel_map_path1_out3_en",
                    value=False,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    "channel_map_path0_out0_en",
                    "channel_map_path1_out1_en",
                    value=False,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    "channel_map_path0_out2_en",
                    "channel_map_path1_out3_en",
                    value=True,
                )

                # on initialisation, disconnect all other sequencers from the ports
                self._device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(2, self._device_num_sequencers):
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out0_en",
                        "channel_map_path1_out1_en",
                        value=False,
                    )  # Default after reboot = True
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out2_en",
                        "channel_map_path1_out3_en",
                        value=False,
                    )  # Default after reboot = True

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters (list): A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument with the settings of the runcard.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['ports']['o1']['channel'] (int | str): the id of the refrigerator channel the output port o1 is connected to.
                kwargs['ports']['o1']['attenuation'] (int): [0 to 60 dBm, in multiples of 2] attenuation at the output.
                kwargs['ports']['o1']['lo_enabled'] (bool): enable or disable local oscillator for up-conversion.
                kwargs['ports']['o1']['lo_frequency'] (int): [2_000_000_000 to 18_000_000_000 Hz] local oscillator frequency.
                kwargs['ports']['o1']['gain'] (float): [0.0 - 1.0 unitless] gain applied prior to up-conversion. Qblox recommends to keep
                    `pulse_amplitude * gain` below 0.3 to ensure the mixers are working in their linear regime, if necessary, lowering the attenuation
                    applied at the output.
                kwargs['ports']['o1']['hardware_mod_en'] (bool): enables Hardware Modulation. In this mode, pulses are modulated to the intermediate frequency
                    using the numerically controlled oscillator within the fpga. It only requires the upload of the pulse envelope waveform.

                kwargs['ports']['o2']['channel'] (int | str): the id of the refrigerator channel the output port o2 is connected to.
                kwargs['ports']['o2']['attenuation'] (int): [0 to 60 dBm, in multiples of 2] attenuation at the output.
                kwargs['ports']['o2']['lo_enabled'] (bool): enable or disable local oscillator for up-conversion.
                kwargs['ports']['o2']['lo_frequency'] (int): [2_000_000_000 to 18_000_000_000 Hz] local oscillator frequency.
                kwargs['ports']['o2']['gain'] (float): [0.0 - 1.0 unitless] gain applied prior to up-conversion. Qblox recommends to keep
                    `pulse_amplitude * gain` below 0.3 to ensure the mixers are working in their linear regime, if necessary, lowering the attenuation
                    applied at the output.
                kwargs['ports']['o2']['hardware_mod_en'] (bool): enables Hardware Modulation. In this mode, pulses are modulated to the intermediate frequency
                    using the numerically controlled oscillator within the fpga. It only requires the upload of the pulse envelope waveform.

        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            for port in ["o1", "o2"]:
                if port in kwargs["ports"]:
                    self.ports[port].channel = kwargs["ports"][port]["channel"]
                    self._port_channel_map[port] = self.ports[port].channel
                    self.ports[port].attenuation = kwargs["ports"][port]["attenuation"]
                    self.ports[port].lo_enabled = kwargs["ports"][port]["lo_enabled"]
                    self.ports[port].lo_frequency = kwargs["ports"][port]["lo_frequency"]
                    self.ports[port].gain = kwargs["ports"][port]["gain"]
                    if "hardware_mod_en" in kwargs["ports"][port]:
                        self.ports[port].hardware_mod_en = kwargs["ports"][port]["hardware_mod_en"]
                    else:
                        self.ports[port].hardware_mod_en = True
                    self.ports[port].nco_freq = 0
                    self.ports[port].nco_phase_offs = 0
                else:
                    if port in self.ports:
                        self.ports[port].attenuation = 60
                        self.ports[port].lo_enabled = False
                        self.ports[port].lo_frequency = 2e9
                        self.ports[port].gain = 0
                        self.ports[port].hardware_mod_en = False
                        self.ports[port].nco_freq = 0
                        self.ports[port].nco_phase_offs = 0
                        self.ports.pop(port)
                        self._output_ports_keys.remove(port)
                        self._sequencers.pop(port)

            self._channel_port_map = {v: k for k, v in self._port_channel_map.items()}
            self.channels = list(self._channel_port_map.keys())

            self._last_pulsequence_hash = 0
        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def _get_next_sequencer(self, port, frequency, qubit: None):
        """Retrieves and configures the next avaliable sequencer.

        The parameters of the new sequencer are copied from those of the default sequencer, except for the
        intermediate frequency and classification parameters.
        Args:
            port (str):
            frequency ():
            qubit ():
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        # select a new sequencer and configure it as required
        next_sequencer_number = self._free_sequencers_numbers.pop(0)
        if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                # exclude read-only parameter `sequence`
                if not parameter in ["sequence"]:
                    value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                    if value:
                        target = self.device.sequencers[next_sequencer_number]
                        self._set_device_parameter(target, parameter, value=value)

        # if hardware modulation is enabled configure nco_frequency
        if self.ports[port].hardware_mod_en:
            self._set_device_parameter(
                self.device.sequencers[next_sequencer_number],
                "nco_freq",
                value=frequency,  # Assumes all pulses in non_overlapping_pulses set
                # have the same frequency. Non-overlapping pulses of different frequencies on the same
                # qubit channel with hardware_demod_en would lead to wrong results.
                # TODO: Throw error in that event or implement for non_overlapping_same_frequency_pulses
            )
        # create sequencer wrapper
        sequencer = Sequencer(next_sequencer_number)
        sequencer.qubit = qubit
        return sequencer

    def get_if(self, pulse):
        """Returns the intermediate frequency needed to synthesise a pulse based on the port lo frequency."""

        _rf = pulse.frequency
        _lo = self.ports[self._channel_port_map[pulse.channel]].lo_frequency
        _if = _rf - _lo
        if abs(_if) > self.FREQUENCY_LIMIT:
            raise RuntimeError(
                f"""
            Pulse frequency {_rf} cannot be synthesised with current lo frequency {_lo}.
            The intermediate frequency {_if} would exceed the maximum frequency of {self.FREQUENCY_LIMIT}
            """
            )
        return _if

    def process_pulse_sequence(
        self, instrument_pulses: PulseSequence, nshots: int, navgs: int, repetition_duration: int, sweepers: list = []
    ):
        """Processes a list of pulses, generating the waveforms and sequence program required by
        the instrument to synthesise them.

        The output of the process is a list of sequencers used for each port, configured with the information
        required to play the sequence.
        The following features are supported:
            - Hardware modulation and demodulation.
            - Multiplexed readout.
            - Sequencer memory optimisation.
            - Extended waveform memory with the use of multiple sequencers.
            - Overlapping pulses.
            - Waveform and Sequence Program cache.
            - Pulses of up to 8192 pairs of i, q samples

        Args:
            instrument_pulses (PulseSequence): A collection of Pulse objects to be played by the instrument.
            nshots (int): The number of times the sequence of pulses should be executed.
            repetition_duration (int): The total duration of the pulse sequence execution plus the reset/relaxation time.
        """

        # Save the hash of the current sequence of pulses.
        self._current_pulsesequence_hash = hash(
            (instrument_pulses, nshots, repetition_duration, (port.hardware_mod_en for port in self.ports))
        )

        # Check if the sequence to be processed is the same as the last one.
        # If so, there is no need to generate new waveforms and program
        if True:  # self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._free_sequencers_numbers = list(range(len(self.ports), 6))

            # process the pulses for every port
            for port in self.ports:
                # split the collection of instruments pulses by ports
                port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self._port_channel_map[port])

                # initialise the list of sequencers required by the port
                self._sequencers[port] = []

                if not port_pulses.is_empty:
                    # initialise the list of free sequencer numbers to include the default for each port {'o1': 0, 'o2': 1}
                    self._free_sequencers_numbers = [self.DEFAULT_SEQUENCERS[port]] + self._free_sequencers_numbers

                    # split the collection of port pulses in non overlapping pulses
                    non_overlapping_pulses: PulseSequence
                    for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                        # TODO: for non_overlapping_same_frequency_pulses in non_overlapping_pulses.separate_different_frequency_pulses():

                        # each set of not overlapping pulses will be played by a separate sequencer
                        # check sequencer availability
                        if len(self._free_sequencers_numbers) == 0:
                            raise Exception(
                                f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                            )

                        # get next sequencer
                        sequencer = self._get_next_sequencer(
                            port=port,
                            frequency=self.get_if(non_overlapping_pulses[0]),
                            qubit=non_overlapping_pulses[0].qubit,
                        )
                        # add the sequencer to the list of sequencers required by the port
                        self._sequencers[port].append(sequencer)

                        # make a temporary copy of the pulses to be processed
                        pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                        while not pulses_to_be_processed.is_empty:
                            pulse: Pulse = pulses_to_be_processed[0]
                            # attempt to save the waveforms to the sequencer waveforms buffer
                            try:
                                sequencer.waveforms_buffer.add_waveforms(pulse, self.ports[port].hardware_mod_en)
                                sequencer.pulses.add(pulse)
                                pulses_to_be_processed.remove(pulse)

                            # if there is not enough memory in the current sequencer, use another one
                            except WaveformsBuffer.NotEnoughMemory:
                                if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                    raise NotImplementedError(
                                        f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported."
                                    )
                                if len(self._free_sequencers_numbers) == 0:
                                    raise Exception(
                                        f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                                    )
                                # get next sequencer
                                sequencer = self._get_next_sequencer(
                                    port=port,
                                    frequency=self.get_if(non_overlapping_pulses[0]),
                                    qubit=non_overlapping_pulses[0].qubit,
                                )
                                # add the sequencer to the list of sequencers required by the port
                                self._sequencers[port].append(sequencer)

            # update the lists of used and unused sequencers that will be needed later on
            self._used_sequencers_numbers = []
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    self._used_sequencers_numbers.append(sequencer.number)
            self._unused_sequencers_numbers = []
            for n in range(self._device_num_sequencers):
                if not n in self._used_sequencers_numbers:
                    self._unused_sequencers_numbers.append(n)

            # generate and store the Waveforms dictionary, the Acquisitions dictionary, the Weights and the Program
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    pulses = sequencer.pulses
                    program = sequencer.program
                    pulse = None
                    for pulse in pulses:
                        pulse.sweeper = None

                    for sweeper in sweepers:
                        reference_value = 0
                        if sweeper.parameter == Parameter.frequency:
                            if sequencer.pulses:
                                reference_value = self.get_if(sequencer.pulses[0])
                        # if sweeper.parameter == Parameter.amplitude:
                        #     reference_value = self.ports[port].gain
                        # if sweeper.parameter == Parameter.bias:
                        #     reference_value = self.ports[port].offset

                        if sweeper.parameter == Parameter.duration and pulse in sweeper.pulses:
                            if pulse in sweeper.pulses:
                                idx_range = sequencer.waveforms_buffer.bake_pulse_waveforms(
                                    pulse, sweeper.values, self.ports[port].hardware_mod_en
                                )
                                sweeper.qs = QbloxSweeper(
                                    program=program, type=QbloxSweeperType.duration, rel_values=idx_range
                                )
                        else:
                            sweeper.qs = QbloxSweeper.from_sweeper(
                                program=program, sweeper=sweeper, add_to=reference_value
                            )

                        # if sweeper.qubits and sequencer.qubit in sweeper.qubits:
                        #     sweeper.qs.update_parameters = True
                        if sweeper.pulses:
                            for pulse in pulses:
                                if pulse in sweeper.pulses:
                                    sweeper.qs.update_parameters = True
                                    pulse.sweeper = sweeper.qs

                    # Waveforms
                    for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                        sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                    # Program
                    minimum_delay_between_instructions = 4

                    sequence_total_duration = pulses.finish  # the minimum delay between instructions is 4ns
                    time_between_repetitions = repetition_duration - sequence_total_duration
                    assert time_between_repetitions > minimum_delay_between_instructions

                    nshots_register = Register(program, "nshots")
                    navgs_register = Register(program, "navgs")
                    bin_n = Register(program, "bin_n")

                    header_block = Block("setup")

                    body_block = Block()

                    body_block.append(f"wait_sync {minimum_delay_between_instructions}")
                    if self.ports[port].hardware_mod_en:
                        body_block.append("reset_ph")
                        body_block.append_spacer()

                    pulses_block = Block("play")
                    # Add an initial wait instruction for the first pulse of the sequence
                    initial_wait_block = wait_block(
                        wait_time=pulses[0].start, register=Register(program), force_multiples_of_4=False
                    )
                    pulses_block += initial_wait_block

                    for n in range(pulses.count):
                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.start:
                            pulses_block.append(f"wait {pulses[n].sweeper.register}")

                        if self.ports[port].hardware_mod_en:
                            # # Set frequency
                            # _if = self.get_if(pulses[n])
                            # pulses_block.append(f"set_freq {convert_frequency(_if)}", f"set intermediate frequency to {_if} Hz")

                            # Set phase
                            pulses_block.append(
                                f"set_ph {convert_phase(pulses[n].relative_phase)}",
                                comment=f"set relative phase {pulses[n].relative_phase} rads",
                            )

                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[n + 1].start - pulses[n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[n].start

                        if delay_after_play < minimum_delay_between_instructions:
                            raise Exception(
                                f"The minimum delay between pulses is {minimum_delay_between_instructions}ns."
                            )

                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.duration:
                            RI = pulses[n].sweeper.register
                            if pulses[n].type == PulseType.FLUX:
                                RQ = pulses[n].sweeper.register
                            else:
                                RQ = pulses[n].sweeper.aux_register

                            pulses_block.append(
                                f"play  {RI},{RQ},{delay_after_play}",  # FIXME delay_after_play won't work as the duration increases
                                comment=f"play pulse {pulses[n]} sweeping its duration",
                            )
                        else:
                            # Prepare play instruction: play wave_i_index, wave_q_index, delay_next_instruction
                            pulses_block.append(
                                f"play  {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}",
                                comment=f"play waveforms {pulses[n]}",
                            )

                    body_block += pulses_block
                    body_block.append_spacer()

                    final_reset_block = wait_block(
                        wait_time=time_between_repetitions, register=Register(program), force_multiples_of_4=False
                    )

                    # final_reset_block.append_spacer()
                    # final_reset_block.append(f"add {bin_n}, 1, {bin_n}", "not used")

                    body_block += final_reset_block

                    footer_block = Block("cleaup")
                    footer_block.append(f"stop")

                    for sweeper in sweepers:
                        # Parameter.bias: sequencer.qubit in sweeper.qubits # + self.ports[port].offset
                        # Parameter.amplitude: sequencer.pulses[0] in sweeper.pulses: # + self.ports[port].gain
                        # Parameter.frequency: sequencer.pulses[0] in sweeper.pulses # + self.get_if(sequencer.pulses[0])

                        body_block = sweeper.qs.block(inner_block=body_block)

                    nshots_block: Block = loop_block(
                        start=0, stop=nshots, step=1, register=nshots_register, block=body_block
                    )

                    # nshots_block.prepend(f"move 0, {bin_n}", "not used")
                    # nshots_block.append_spacer()

                    navgs_block = loop_block(start=0, stop=navgs, step=1, register=navgs_register, block=nshots_block)
                    program.add_blocks(header_block, navgs_block, footer_block)

                    sequencer.program = repr(program)

    def upload(self):
        """Uploads waveforms and programs of all sequencers and arms them in preparation for execution.

        This method should be called after `process_pulse_sequence()`.
        It configures certain parameters of the instrument based on the needs of resources determined
        while processing the pulse sequence.
        """
        if True:  # self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._last_pulsequence_hash = self._current_pulsesequence_hash

            # Setup
            for sequencer_number in self._used_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=True)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0

            for sequencer_number in self._unused_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=False)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=0)  # Default after reboot = 0
                if sequencer_number >= 2:  # Never disconnect default sequencers
                    self._set_device_parameter(target, "channel_map_path0_out0_en", value=False)
                    self._set_device_parameter(target, "channel_map_path0_out2_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out1_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out3_en", value=False)

            # Upload waveforms and program
            qblox_dict = {}
            sequencer: Sequencer
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    # Add sequence program and waveforms to single dictionary
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program,
                    }

                    # Upload dictionary to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(qblox_dict[sequencer])

                    # DEBUG: QCM RF Save sequence to file
                    filename = f"Z_{self.name}_sequencer{sequencer.number}_sequence.json"
                    with open(filename, "w", encoding="utf-8") as file:
                        json.dump(qblox_dict[sequencer], file, indent=4)
                        file.write(sequencer.program)

        # Arm sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.arm_sequencer(sequencer_number)

        # DEBUG: QCM RF Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

        # DEBUG: QCM RF Save Readable Snapshot
        # filename = f"Z_{self.name}_snapshot.json"
        # with open(filename, "w", encoding="utf-8") as file:
        #     print_readable_snapshot(self.device, file, update=True)

    def play_sequence(self):
        """Plays the sequence of pulses.

        Starts the sequencers needed to play the sequence of pulses."""
        for sequencer_number in self._used_sequencers_numbers:
            # Start used sequencers
            self.device.start_sequencer(sequencer_number)
            # DEBUG sync_en
            # print(
            #     f"device: {self.name}, sequencer: {sequencer_number}, sync_en: {self.device.sequencers[sequencer_number].get('sync_en')}"
            # )

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Stops all sequencers"""
        try:
            self.device.stop_sequencer()
        except:
            pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        self._cluster = None
        self.is_connected = False


class ClusterQCM(AbstractInstrument):
    """Qblox Cluster Qubit Control Module Baseband driver.

    Qubit Control Module (QCM) is an arbitratry wave generator with two DACs connected to
    four output ports. It can sinthesise either four independent real signals or two
    complex signals, using ports 0 and 2 to output the i(in-phase) component and
    ports 1 and 3 the q(quadrature) component. The sampling rate of its DAC is 1 GSPS.
    https://www.qblox.com/cluster

    The class aims to simplify the configuration of the instrument, exposing only
    those parameters most frequencly used and hiding other more complex components.

    A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QRM_QCM`
    object is provided via the attribute `device`, allowing the advanced user to gain
    access to the features that are not exposed directly by the class.

    In order to accelerate the execution, the instrument settings are cached, so that
    the communication with the instrument only happens when the parameters change.
    This caching is done with the method `_set_device_parameter(target, *parameters, value)`.

            ports:
                o1:
                    channel                      : L4-1
                    gain                         : 0.2 # -1.0<=v<=1.0
                    offset                       : 0   # -2.5<=v<=2.5
                o2:
                    channel                      : L4-2
                    gain                         : 0.2 # -1.0<=v<=1.0
                    offset                       : 0   # -2.5<=v<=2.5
                o3:
                    channel                      : L4-3
                    gain                         : 0.2 # -1.0<=v<=1.0
                    offset                       : 0   # -2.5<=v<=2.5
                o4:
                    channel                      : L4-4
                    gain                         : 0.2 # -1.0<=v<=1.0
                    offset                       : 0   # -2.5<=v<=2.5

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        name (str): A unique name given to the instrument.
        address (str): IP_address:module_number (the IP address of the cluster and
            the module number)
        device (QbloxQrmQcm): A reference to the underlying
            `qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm` object. It can be used to access other
            features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html

        ports = A dictionary giving access to the output ports objects.
            ports['o1']
            ports['o2']
            ports['o3']
            ports['o4']

            ports['oX'].channel (int | str): the id of the refrigerator channel the port is connected to.
            ports['oX'].gain (float): (mapped to qrm.sequencers[0].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['oX'].offset (float): (mapped to qrm.outX_offset)
                Sets the offset on the output port.
            ports['oX'].hardware_mod_en (bool): (mapped to qrm.sequencers[0].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['oX'].nco_freq (int): (mapped to qrm.sequencers[0].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['oX'].nco_phase_offs = (mapped to qrm.sequencers[0].nco_phase_offs) # TODO mapped, but not configurable from the runcard

        Sequencer 0 is always the first sequencer used to synthesise pulses on port o1.
        Sequencer 1 is always the first sequencer used to synthesise pulses on port o2.
        Sequencer 2 is always the first sequencer used to synthesise pulses on port o3.
        Sequencer 3 is always the first sequencer used to synthesise pulses on port o4.
        Sequencer 4 to 6 are used as needed to sinthesise simultaneous pulses on the same channel
        or when the memory of the default sequencers rans out.

    """

    DEFAULT_SEQUENCERS = {"o1": 0, "o2": 1, "o3": 2, "o4": 3}
    SAMPLING_RATE: int = 1e9  # 1 GSPS
    FREQUENCY_LIMIT = 500e6

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device.sequencers[sequencer], *parameter, value=x),
    )

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        """Initialises the instance.

        All class attributes are defined and initialised.
        """
        self.device: QbloxQrmQcm = None
        self.ports: dict = {}
        self.channels: list = []

        self._cluster: QbloxCluster = None
        self._output_ports_keys = ["o1", "o2", "o3", "o4"]
        self._sequencers: dict[Sequencer] = {"o1": [], "o2": [], "o3": [], "o4": []}
        self._port_channel_map: dict = {}
        self._channel_port_map: dict = {}
        self._last_pulsequence_hash: int = 0
        self._current_pulsesequence_hash: int
        self._device_parameters = {}
        self._device_num_output_ports = 2
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []

    def connect(self):
        """Connects to the instrument using the instrument settings in the runcard.

        Once connected, it creates port classes with properties mapped to various instrument
        parameters, and initialises the the underlying device parameters.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device = cluster.modules[int(self.address.split(":")[1]) - 1]
                for n in range(4):
                    port = "o" + str(n + 1)
                    self.ports[port] = type(
                        f"port_" + port,
                        (),
                        {
                            "channel": None,
                            "gain": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS[port], "gain_awg_path0"),
                            "offset": self.property_wrapper(f"out{n}_offset"),
                            "hardware_mod_en": self.sequencer_property_wrapper(
                                self.DEFAULT_SEQUENCERS[port], "mod_en_awg"
                            ),
                            "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS[port], "nco_freq"),
                            "nco_phase_offs": self.sequencer_property_wrapper(
                                self.DEFAULT_SEQUENCERS[port], "nco_phase_offs"
                            ),
                            "qubit": None,
                        },
                    )()

                # save reference to cluster
                self._cluster = cluster
                self.is_connected = True

                # once connected, initialise the parameters of the device to the default values
                self._set_device_parameter(self.device, "out0_offset", value=0)  # Default after reboot = 0
                self._set_device_parameter(self.device, "out1_offset", value=0)  # Default after reboot = 0
                self._set_device_parameter(self.device, "out2_offset", value=0)  # Default after reboot = 0
                self._set_device_parameter(self.device, "out3_offset", value=0)  # Default after reboot = 0

                for target in [
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o3"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o4"]],
                ]:
                    self._set_device_parameter(
                        target, "cont_mode_en_awg_path0", "cont_mode_en_awg_path1", value=False
                    )  # Default after reboot = False
                    self._set_device_parameter(
                        target, "cont_mode_waveform_idx_awg_path0", "cont_mode_waveform_idx_awg_path1", value=0
                    )  # Default after reboot = 0
                    self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                    self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0
                    self._set_device_parameter(target, "mixer_corr_gain_ratio", value=1)
                    self._set_device_parameter(target, "mixer_corr_phase_offset_degree", value=0)
                    self._set_device_parameter(target, "offset_awg_path0", value=0)
                    self._set_device_parameter(target, "offset_awg_path1", value=0)
                    self._set_device_parameter(target, "sync_en", value=False)  # Default after reboot = False
                    self._set_device_parameter(target, "upsample_rate_awg_path0", "upsample_rate_awg_path1", value=0)

                for target in [
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o3"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o4"]],
                ]:
                    self._set_device_parameter(
                        target, "channel_map_path0_out0_en", value=False
                    )  # Default after reboot = True
                    self._set_device_parameter(
                        target, "channel_map_path1_out1_en", value=False
                    )  # Default after reboot = True
                    self._set_device_parameter(
                        target, "channel_map_path0_out2_en", value=False
                    )  # Default after reboot = True
                    self._set_device_parameter(
                        target, "channel_map_path1_out3_en", value=False
                    )  # Default after reboot = True

                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    "channel_map_path0_out0_en",
                    value=True,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    "channel_map_path1_out1_en",
                    value=True,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o3"]],
                    "channel_map_path0_out2_en",
                    value=True,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o4"]],
                    "channel_map_path1_out3_en",
                    value=True,
                )

                # on initialisation, disconnect all other sequencers from the ports
                self._device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(4, self._device_num_sequencers):
                    target = self.device.sequencers[sequencer]
                    self._set_device_parameter(target, "channel_map_path0_out0_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out1_en", value=False)
                    self._set_device_parameter(target, "channel_map_path0_out2_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out3_en", value=False)

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters (list): A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument with the settings of the runcard.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                oX: ['o1', 'o2', 'o3', 'o4']
                kwargs['ports']['oX']['channel'] (int | str): the id of the refrigerator channel the port is connected to.
                kwargs['ports'][oX]['gain'] (float): [0.0 - 1.0 unitless] gain applied prior to up-conversion. Qblox recommends to keep
                    `pulse_amplitude * gain` below 0.3 to ensure the mixers are working in their linear regime, if necessary, lowering the attenuation
                    applied at the output.
                kwargs['ports'][oX]['offset'] (float): [-2.5 - 2.5 V] offset in volts applied to the output port.
                kwargs['ports'][oX]['hardware_mod_en'] (bool): enables Hardware Modulation. In this mode, pulses are modulated to the intermediate frequency
                    using the numerically controlled oscillator within the fpga. It only requires the upload of the pulse envelope waveform.

        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            for port in ["o1", "o2", "o3", "o4"]:
                if port in kwargs["ports"]:
                    self.ports[port].channel = kwargs["ports"][port]["channel"]
                    self._port_channel_map[port] = self.ports[port].channel
                    self.ports[port].gain = kwargs["ports"][port]["gain"]
                    self.ports[port].offset = kwargs["ports"][port]["offset"]
                    if "hardware_mod_en" in kwargs["ports"][port]:
                        self.ports[port].hardware_mod_en = kwargs["ports"][port]["hardware_mod_en"]
                    else:
                        self.ports[port].hardware_mod_en = True
                    self.ports[port].qubit = kwargs["ports"][port]["qubit"]
                    self.ports[port].nco_freq = 0
                    self.ports[port].nco_phase_offs = 0
                else:
                    if port in self.ports:
                        self.ports[port].channel = None
                        self.ports[port].gain = 0
                        self.ports[port].offset = 0
                        self.ports[port].hardware_mod_en = False
                        self.ports[port].qubit = None
                        self.ports[port].nco_freq = 0
                        self.ports[port].nco_phase_offs = 0
                        self.ports.pop(port)
                        self._output_ports_keys.remove(port)
                        self._sequencers.pop(port)

            self._channel_port_map = {v: k for k, v in self._port_channel_map.items()}
            self.channels = list(self._channel_port_map.keys())

            self._last_pulsequence_hash = 0
        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def _get_next_sequencer(self, port, frequency, qubit: None):
        """Retrieves and configures the next avaliable sequencer.

        The parameters of the new sequencer are copied from those of the default sequencer, except for the
        intermediate frequency and classification parameters.
        Args:
            port (str):
            frequency ():
            qubit ():
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        # select a new sequencer and configure it as required
        next_sequencer_number = self._free_sequencers_numbers.pop(0)
        if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                # exclude read-only parameter `sequence`
                if not parameter in ["sequence"]:
                    value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                    if value:
                        target = self.device.sequencers[next_sequencer_number]
                        self._set_device_parameter(target, parameter, value=value)

        # if hardware modulation is enabled configure nco_frequency
        if self.ports[port].hardware_mod_en:
            self._set_device_parameter(
                self.device.sequencers[next_sequencer_number],
                "nco_freq",
                value=frequency,  # Assumes all pulses in non_overlapping_pulses set
                # have the same frequency. Non-overlapping pulses of different frequencies on the same
                # qubit channel with hardware_demod_en would lead to wrong results.
                # TODO: Throw error in that event or implement for non_overlapping_same_frequency_pulses
            )
        # create sequencer wrapper
        sequencer = Sequencer(next_sequencer_number)
        sequencer.qubit = qubit
        return sequencer

    def get_if(self, pulse):
        """Returns the intermediate frequency needed to synthesise a pulse based on the port lo frequency."""

        _rf = pulse.frequency
        _lo = 0
        _if = _rf - _lo
        if abs(_if) > self.FREQUENCY_LIMIT:
            raise RuntimeError(
                f"""
            Pulse frequency {_rf} cannot be synthesised with current lo frequency {_lo}.
            The intermediate frequency {_if} would exceed the maximum frequency of {self.FREQUENCY_LIMIT}
            """
            )
        return _if

    def process_pulse_sequence(
        self, instrument_pulses: PulseSequence, nshots: int, navgs: int, repetition_duration: int, sweepers: list = []
    ):
        """Processes a list of pulses, generating the waveforms and sequence program required by
        the instrument to synthesise them.

        The output of the process is a list of sequencers used for each port, configured with the information
        required to play the sequence.
        The following features are supported:
            - Hardware modulation and demodulation.
            - Multiplexed readout.
            - Sequencer memory optimisation.
            - Extended waveform memory with the use of multiple sequencers.
            - Overlapping pulses.
            - Waveform and Sequence Program cache.
            - Pulses of up to 8192 pairs of i, q samples

        Args:
            instrument_pulses (PulseSequence): A collection of Pulse objects to be played by the instrument.
            nshots (int): The number of times the sequence of pulses should be executed.
            repetition_duration (int): The total duration of the pulse sequence execution plus the reset/relaxation time.
        """

        # Save the hash of the current sequence of pulses.
        self._current_pulsesequence_hash = hash(
            (instrument_pulses, nshots, repetition_duration, (port.hardware_mod_en for port in self.ports))
        )

        # Check if the sequence to be processed is the same as the last one.
        # If so, there is no need to generate new waveforms and program
        if True:  # self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._free_sequencers_numbers = list(range(len(self.ports), 6))

            # process the pulses for every port
            for port in self.ports:
                # split the collection of instruments pulses by ports
                port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self._port_channel_map[port])

                # initialise the list of sequencers required by the port
                self._sequencers[port] = []

                # initialise the list of free sequencer numbers to include the default for each port {'o1': 0, 'o2': 1, 'o3': 2, 'o4': 3}
                self._free_sequencers_numbers = [self.DEFAULT_SEQUENCERS[port]] + self._free_sequencers_numbers

                if not port_pulses.is_empty:
                    # split the collection of port pulses in non overlapping pulses
                    non_overlapping_pulses: PulseSequence
                    for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                        # TODO: for non_overlapping_same_frequency_pulses in non_overlapping_pulses.separate_different_frequency_pulses():

                        # each set of not overlapping pulses will be played by a separate sequencer
                        # check sequencer availability
                        if len(self._free_sequencers_numbers) == 0:
                            raise Exception(
                                f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                            )
                        # get next sequencer
                        sequencer = self._get_next_sequencer(
                            port=port,
                            frequency=self.get_if(non_overlapping_pulses[0]),
                            qubit=non_overlapping_pulses[0].qubit,
                        )
                        # add the sequencer to the list of sequencers required by the port
                        self._sequencers[port].append(sequencer)

                        # make a temporary copy of the pulses to be processed
                        pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                        while not pulses_to_be_processed.is_empty:
                            pulse: Pulse = pulses_to_be_processed[0]
                            # attempt to save the waveforms to the sequencer waveforms buffer
                            try:
                                sequencer.waveforms_buffer.add_waveforms(pulse, self.ports[port].hardware_mod_en)
                                sequencer.pulses.add(pulse)
                                pulses_to_be_processed.remove(pulse)

                            # if there is not enough memory in the current sequencer, use another one
                            except WaveformsBuffer.NotEnoughMemory:
                                if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                    raise NotImplementedError(
                                        f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported."
                                    )
                                if len(self._free_sequencers_numbers) == 0:
                                    raise Exception(
                                        f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                                    )
                                # get next sequencer
                                sequencer = self._get_next_sequencer(
                                    port=port,
                                    frequency=self.get_if(non_overlapping_pulses[0]),
                                    qubit=non_overlapping_pulses[0].qubit,
                                )
                                # add the sequencer to the list of sequencers required by the port
                                self._sequencers[port].append(sequencer)
                else:
                    sequencer = self._get_next_sequencer(port=port, frequency=0, qubit=self.ports[port].qubit)
                    # add the sequencer to the list of sequencers required by the port
                    self._sequencers[port].append(sequencer)

            # update the lists of used and unused sequencers that will be needed later on
            self._used_sequencers_numbers = []
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    self._used_sequencers_numbers.append(sequencer.number)
            self._unused_sequencers_numbers = []
            for n in range(self._device_num_sequencers):
                if not n in self._used_sequencers_numbers:
                    self._unused_sequencers_numbers.append(n)

            # generate and store the Waveforms dictionary, the Acquisitions dictionary, the Weights and the Program
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    pulses = sequencer.pulses
                    program = sequencer.program
                    pulse = None
                    for pulse in pulses:
                        pulse.sweeper = None

                    for sweeper in sweepers:
                        reference_value = 0
                        if sweeper.parameter == Parameter.frequency:
                            if sequencer.pulses:
                                reference_value = self.get_if(sequencer.pulses[0])
                        # if sweeper.parameter == Parameter.amplitude:
                        #     reference_value = self.ports[port].gain
                        # if sweeper.parameter == Parameter.bias: (this goes on top of the external offset)
                        #     reference_value = self.ports[port].offset

                        if sweeper.parameter == Parameter.duration and pulse in sweeper.pulses:
                            if pulse in sweeper.pulses:
                                idx_range = sequencer.waveforms_buffer.bake_pulse_waveforms(
                                    pulse, sweeper.values, self.ports[port].hardware_mod_en
                                )
                                sweeper.qs = QbloxSweeper(
                                    program=program, type=QbloxSweeperType.duration, rel_values=idx_range
                                )
                        else:
                            sweeper.qs = QbloxSweeper.from_sweeper(
                                program=program, sweeper=sweeper, add_to=reference_value
                            )

                        if sweeper.qubits and sequencer.qubit in sweeper.qubits:
                            sweeper.qs.update_parameters = True
                        if sweeper.pulses:
                            for pulse in pulses:
                                if pulse in sweeper.pulses:
                                    sweeper.qs.update_parameters = True
                                    pulse.sweeper = sweeper.qs

                    # Waveforms
                    for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                        sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                    # Program
                    minimum_delay_between_instructions = 4

                    sequence_total_duration = pulses.finish  # the minimum delay between instructions is 4ns
                    time_between_repetitions = repetition_duration - sequence_total_duration
                    assert time_between_repetitions > minimum_delay_between_instructions

                    nshots_register = Register(program, "nshots")
                    navgs_register = Register(program, "navgs")
                    bin_n = Register(program, "bin_n")

                    header_block = Block("setup")

                    body_block = Block()

                    body_block.append(f"wait_sync {minimum_delay_between_instructions}")
                    if self.ports[port].hardware_mod_en:
                        body_block.append("reset_ph")
                        body_block.append_spacer()

                    pulses_block = Block("play")
                    # Add an initial wait instruction for the first pulse of the sequence
                    initial_wait_block = wait_block(
                        wait_time=pulses.start, register=Register(program), force_multiples_of_4=False
                    )
                    pulses_block += initial_wait_block

                    for n in range(pulses.count):
                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.start:
                            pulses_block.append(f"wait {pulses[n].sweeper.register}")

                        if self.ports[port].hardware_mod_en:
                            # # Set frequency
                            # _if = self.get_if(pulses[n])
                            # pulses_block.append(f"set_freq {convert_frequency(_if)}", f"set intermediate frequency to {_if} Hz")

                            # Set phase
                            pulses_block.append(
                                f"set_ph {convert_phase(pulses[n].relative_phase)}",
                                comment=f"set relative phase {pulses[n].relative_phase} rads",
                            )

                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[n + 1].start - pulses[n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[n].start

                        if delay_after_play < minimum_delay_between_instructions:
                            raise Exception(
                                f"The minimum delay between pulses is {minimum_delay_between_instructions}ns."
                            )

                        if pulses[n].sweeper and pulses[n].sweeper.type == QbloxSweeperType.duration:
                            RI = pulses[n].sweeper.register
                            if pulses[n].type == PulseType.FLUX:
                                RQ = pulses[n].sweeper.register
                            else:
                                RQ = pulses[n].sweeper.aux_register

                            pulses_block.append(
                                f"play  {RI},{RQ},{delay_after_play}",  # FIXME delay_after_play won't work as the duration increases
                                comment=f"play pulse {pulses[n]} sweeping its duration",
                            )
                        else:
                            # Prepare play instruction: play wave_i_index, wave_q_index, delay_next_instruction
                            pulses_block.append(
                                f"play  {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}",
                                comment=f"play waveforms {pulses[n]}",
                            )

                    body_block += pulses_block
                    body_block.append_spacer()

                    final_reset_block = wait_block(
                        wait_time=time_between_repetitions, register=Register(program), force_multiples_of_4=False
                    )
                    body_block += final_reset_block

                    footer_block = Block("cleaup")
                    footer_block.append(f"stop")

                    for sweeper in sweepers:
                        # Parameter.bias: sequencer.qubit in sweeper.qubits # + self.ports[port].offset
                        # Parameter.amplitude: sequencer.pulses[0] in sweeper.pulses: # + self.ports[port].gain
                        # Parameter.frequency: sequencer.pulses[0] in sweeper.pulses # + self.get_if(sequencer.pulses[0])

                        body_block = sweeper.qs.block(inner_block=body_block)

                    nshots_block: Block = loop_block(
                        start=0, stop=nshots, step=1, register=nshots_register, block=body_block
                    )
                    navgs_block = loop_block(start=0, stop=navgs, step=1, register=navgs_register, block=nshots_block)
                    program.add_blocks(header_block, navgs_block, footer_block)

                    sequencer.program = repr(program)

    def upload(self):
        """Uploads waveforms and programs of all sequencers and arms them in preparation for execution.

        This method should be called after `process_pulse_sequence()`.
        It configures certain parameters of the instrument based on the needs of resources determined
        while processing the pulse sequence.
        """
        if True:  # self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._last_pulsequence_hash = self._current_pulsesequence_hash

            # Setup
            for sequencer_number in self._used_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=True)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0

            for sequencer_number in self._unused_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=False)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=0)  # Default after reboot = 0
                if sequencer_number >= 4:  # Never disconnect default sequencers
                    self._set_device_parameter(target, "channel_map_path0_out0_en", value=False)
                    self._set_device_parameter(target, "channel_map_path0_out2_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out1_en", value=False)
                    self._set_device_parameter(target, "channel_map_path1_out3_en", value=False)

            # There seems to be a bug in qblox that when any of the mappings between paths and outputs is set,
            # the general offset goes to 0 (eventhout the parameter will still show the right value).
            # Until that is fixed, I'm going to always set the offset just before playing (bypassing the cache):
            self.device.out0_offset(self.device.get("out0_offset"))
            self.device.out1_offset(self.device.get("out1_offset"))
            self.device.out2_offset(self.device.get("out2_offset"))
            self.device.out3_offset(self.device.get("out3_offset"))

            # Upload waveforms and program
            qblox_dict = {}
            sequencer: Sequencer
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    # Add sequence program and waveforms to single dictionary and write to JSON file
                    filename = f"Z_{self.name}_sequencer{sequencer.number}_sequence.json"
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program,
                    }

                    # Upload dictionary to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(qblox_dict[sequencer])

                    # DEBUG: QCM Save sequence to file
                    # filename = f"Z_{self.name}_sequencer{sequencer.number}_sequence.json"
                    # with open(filename, "w", encoding="utf-8") as file:
                    #     json.dump(qblox_dict[sequencer], file, indent=4)
                    #     file.write(sequencer.program)

        # Arm sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.arm_sequencer(sequencer_number)

        # DEBUG: QCM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

        # DEBUG: QCM Save Readable Snapshot
        # filename = f"Z_{self.name}_snapshot.json"
        # with open(filename, "w", encoding="utf-8") as file:
        #     print_readable_snapshot(self.device, file, update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""

        for sequencer_number in self._used_sequencers_numbers:
            # Start used sequencers
            self.device.start_sequencer(sequencer_number)

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Stops all sequencers"""
        try:
            self.device.stop_sequencer()
        except:
            pass

        try:
            self._set_device_parameter(self.device, "out0_offset", value=0)
            self._set_device_parameter(self.device, "out1_offset", value=0)
            self._set_device_parameter(self.device, "out2_offset", value=0)
            self._set_device_parameter(self.device, "out3_offset", value=0)
            # self.device.out0_offset(0)
            # self.device.out1_offset(0)
            # self.device.out2_offset(0)
            # self.device.out3_offset(0)
        except:
            pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        self._cluster = None
        self.is_connected = False


def print_readable_snapshot(device, file, update: bool = False, max_chars: int = 80) -> None:
    """
    Prints a readable version of the snapshot.
    The readable snapshot includes the name, value and unit of each
    parameter.
    A convenience function to quickly get an overview of the
    status of an instrument.

    Args:
        update: If ``True``, update the state by querying the
            instrument. If ``False``, just use the latest values in memory.
            This argument gets passed to the snapshot function.
        max_chars: the maximum number of characters per line. The
            readable snapshot will be cropped if this value is exceeded.
            Defaults to 80 to be consistent with default terminal width.
    """
    floating_types = (float, np.integer, np.floating)
    snapshot = device.snapshot(update=update)

    par_lengths = [len(p) for p in snapshot["parameters"]]

    # Min of 50 is to prevent a super long parameter name to break this
    # function
    par_field_len = min(max(par_lengths) + 1, 50)

    file.write(device.name + ":" + "\n")
    file.write("{0:<{1}}".format("\tparameter ", par_field_len) + "value" + "\n")
    file.write("-" * max_chars + "\n")
    for par in sorted(snapshot["parameters"]):
        name = snapshot["parameters"][par]["name"]
        msg = "{0:<{1}}:".format(name, par_field_len)

        # in case of e.g. ArrayParameters, that usually have
        # snapshot_value == False, the parameter may not have
        # a value in the snapshot
        val = snapshot["parameters"][par].get("value", "Not available")

        unit = snapshot["parameters"][par].get("unit", None)
        if unit is None:
            # this may be a multi parameter
            unit = snapshot["parameters"][par].get("units", None)
        if isinstance(val, floating_types):
            msg += f"\t{val:.5g} "
            # numpy float and int types format like builtins
        else:
            msg += f"\t{val} "
        if unit != "":  # corresponds to no unit
            msg += f"({unit})"
        # Truncate the message if it is longer than max length
        if len(msg) > max_chars and not max_chars == -1:
            msg = msg[0 : max_chars - 3] + "..."
        file.write(msg + "\n")

    for submodule in device.submodules.values():
        print_readable_snapshot(submodule, file, update=update, max_chars=max_chars)
