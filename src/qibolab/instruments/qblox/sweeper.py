from enum import Enum, auto

import numpy as np

from qibolab.instruments.qblox.q1asm import (
    Block,
    Program,
    Register,
    convert_frequency,
    convert_gain,
    convert_offset,
    convert_phase,
)
from qibolab.sweeper import Parameter, Sweeper


class QbloxSweeperType(Enum):
    """An enumeration for the different types of sweepers supported by qblox.

    - frequency: sweeps pulse frequency by adjusting the sequencer `nco_freq` with q1asm command `set_freq`.
    - gain: sweeps sequencer gain by adjusting the sequencer `gain_awg_path0` and `gain_awg_path1` with q1asm command
      `set_awg_gain`. Since the gain is a parameter between -1 and 1 that multiplies the samples of the waveforms
      before they are fed to the DACs, it can be used to sweep the pulse amplitude.
    - offset: sweeps sequencer offset by adjusting the sequencer `offset_awg_path0` and `offset_awg_path1` with q1asm
      command `set_awg_offs`
    - start: sweeps pulse start.
    - duration: sweeps pulse duration.
    """

    frequency = auto()
    gain = auto()
    offset = auto()
    start = auto()
    duration = auto()

    number = auto()  # internal
    relative_phase = auto()  # not implemented yet
    time = auto()  # not implemented yet


class QbloxSweeper:
    """A custom sweeper object with the data and functionality required by
    qblox instruments.

    It is responsible for generating the q1asm code required to execute sweeps in a sequencer. The object can be
    initialised with either:

    - a :class:`qibolab.sweepers.Sweeper` using the :func:`qibolab.instruments.qblox.QbloxSweeper.from_sweeper`, or
    - a range of values and a sweeper type (:class:`qibolab.instruments.qblox.QbloxSweeperType`)

    Like most FPGAs, qblox FPGAs do not support floating point arithmetics. All parameters that can be manipulated in
    real time within the FPGA are represented as two's complement integers.

    Attributes:
        type (:class:`qibolab.instruments.qblox.QbloxSweeperType`): the type of sweeper
        name (str): a name given for the sweep that is later used within the q1asm code to identify the loops.
        register (:class:`qibolab.instruments.qblox_q1asm.Register`): the main Register (q1asm variable) used in the loop.
        aux_register (:class:`qibolab.instruments.qblox_q1asm.Register`): an auxialiry Register requried in duration
            sweeps.
        update_parameters (Bool): a flag to instruct the sweeper to update the paramters or not depending on whether
            a parameter of the sequencer needs to be swept or not.

    Methods:
        block(inner_block: :class:`qibolab.instruments.qblox_q1asm.Block`): generates the block of q1asm code that implements
            the sweep.
    """

    FREQUENCY_LIMIT = 500e6

    def __init__(
        self,
        program: Program,
        rel_values: list,
        type: QbloxSweeperType = QbloxSweeperType.number,
        add_to: float = 0,
        multiply_to: float = 1,
        name: str = "",
    ):
        """Creates an instance from a range of values and a sweeper type
        (:class:`qibolab.instruments.qblox.QbloxSweeperType`).

        Args:
            program (:class:`qibolab.instruments.qblox_q1asm.Program`): a program object representing the q1asm program
                of a sequencer.
            rel_values (list): a list of values to iterate over. Currently qblox only supports a list of equally spaced
                values, like those created with `np.arange(start, stop, step)`. These values are considered relative
                values. They will later be added to the `add_to` parameter and multiplied to the `multiply_to`
                parameter.
            type (:class:`qibolab.instruments.qblox.QbloxSweeperType`): the type of sweeper.
            add_to (float): a value to be added to each value of the range of values defined in `sweeper.values` or
                `rel_values`.
            multiply_to (float): a value to be multiplied by each value of the range of values defined in
            `sweeper.values` or `rel_values`.
            name (str): a name given for the sweep that is later used within the q1as m code to identify the loops.
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

        # Converted values (converted to q1asm values, two's complement)
        self._con_start: int = None
        self._con_step: int = None
        self._con_stop: int = None
        self._con_values: np.ndarray = None

        # Validate input parameters
        if not len(rel_values) > 1:
            raise ValueError("values must contain at least 2 elements.")
        elif rel_values[1] == rel_values[0]:
            raise ValueError("values must contain different elements.")

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
                lambda v: all(
                    (-self.FREQUENCY_LIMIT <= x and x <= self.FREQUENCY_LIMIT)
                    for x in v
                )
            ),
            QbloxSweeperType.gain: (lambda v: all((-1 <= x and x <= 1) for x in v)),
            QbloxSweeperType.offset: (
                lambda v: all(
                    (-1.25 * np.sqrt(2) <= x and x <= 1.25 * np.sqrt(2)) for x in v
                )
            ),
            QbloxSweeperType.relative_phase: (lambda v: True),
            QbloxSweeperType.start: (lambda v: all((4 <= x and x < 2**16) for x in v)),
            QbloxSweeperType.duration: (
                lambda v: all((0 <= x and x < 2**16) for x in v)
            ),
            QbloxSweeperType.number: (
                lambda v: all((-(2**16) < x and x < 2**16) for x in v)
            ),
        }

        if not check_values[type](np.append(self._abs_values, [self._abs_stop])):
            raise ValueError(
                f"Sweeper {self.name} values are not within the allowed range"
            )

        # Convert absolute values to q1asm values
        convert = {
            QbloxSweeperType.frequency: convert_frequency,
            QbloxSweeperType.gain: convert_gain,
            QbloxSweeperType.offset: convert_offset,
            QbloxSweeperType.relative_phase: convert_phase,
            QbloxSweeperType.start: (lambda x: int(x) % 2**16),
            QbloxSweeperType.duration: (lambda x: int(x) % 2**16),
            QbloxSweeperType.number: (lambda x: int(x) % 2**32),
        }

        self._con_start = convert[type](self._abs_start)
        self._con_step = convert[type](self._abs_step)
        self._con_stop = (self._con_start + self._con_step * (self._n) + 1) % 2**32
        self._con_values = np.array(
            [(self._con_start + self._con_step * m) % 2**32 for m in range(self._n + 1)]
        )

        # log.info(f"Qblox sweeper converted values: {self._con_values}")

        if not (
            isinstance(self._con_start, int)
            and isinstance(self._con_stop, int)
            and isinstance(self._con_step, int)
        ):
            raise ValueError("start, stop and step must be int")

    @classmethod
    def from_sweeper(
        cls,
        program: Program,
        sweeper: Sweeper,
        add_to: float = 0,
        multiply_to: float = 1,
        name: str = "",
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
            Parameter.relative_phase: QbloxSweeperType.relative_phase,
        }
        if sweeper.parameter in type_c:
            type = type_c[sweeper.parameter]
            rel_values = sweeper.values
        else:
            raise ValueError(
                f"Sweeper parameter {sweeper.parameter} is not supported by qblox driver yet."
            )
        return cls(
            program=program,
            rel_values=rel_values,
            type=type,
            add_to=add_to,
            multiply_to=multiply_to,
            name=name,
        )

    def block(self, inner_block: Block):
        """Generates the block of q1asm code that implements the sweep.

        The q1asm code for a sweeper has the following structure:

        .. code-block:: text

            # header_block
            # initialise register with start value
            move    0, R0           # 0 = start value, R0 = register name
            nop                     # wait an instruction cycle (4ns) for the register to be updated with its value
            loop_R0:                # loop label

                # update_parameter_block
                # update parameters, in this case pulse frequency
                set_freq    R0      # sets the frequency of the sequencer nco to the value stored in R0
                upd_param   100     # makes the change effective and wait 100ns

                # inner block
                play 0,1,4          # play waveforms with index 0 and 1 (i and q) and wait 4ns

            # footer_block
            # increment or decrement register with step value
            add R0, 2500, R0        # R0 = R0 + 2500
            nop                     # wait an instruction cycle (4ns) for the register to be updated with its value
            # check condition and loop
            jlt R0, 10001, @loop_R0 # while R0 is less than the stop value loop to loop_R0
                                    # in this example it would loop 5 times
                                    # with R0 values of 0, 2500, 5000, 7500 and 10000

        Args:
            inner_block (:class:`qibolab.instruments.qblox_q1asm.Block`): the block of q1asm code to be repeated within
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
                update_parameter_block.append(
                    f"set_freq {self.register}"
                )  # TODO: move to pulse
                update_parameter_block.append(f"upd_param {update_time}")
            if self.type == QbloxSweeperType.gain:
                update_parameter_block.append(
                    f"set_awg_gain {self.register}, {self.register}"
                )  # TODO: move to pulse
                update_parameter_block.append(f"upd_param {update_time}")
            if self.type == QbloxSweeperType.offset:
                update_parameter_block.append(
                    f"set_awg_offs {self.register}, {self.register}"
                )
                update_parameter_block.append(f"upd_param {update_time}")

            if self.type == QbloxSweeperType.start:
                pass
            if self.type == QbloxSweeperType.duration:
                update_parameter_block.append(
                    f"add {self.register}, 1, {self.aux_register}"
                )
            if self.type == QbloxSweeperType.time:
                pass
            if self.type == QbloxSweeperType.number:
                pass
            if self.type == QbloxSweeperType.relative_phase:
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

        # Qblox fpgas implement negative numbers using two's complement however their conditional jump instructions
        # (jlt and jge) only work with unsigned integers. Negative numbers (from 2**31 to 2**32) are greater than
        # possitive numbers (0 to 2**31). There is therefore a discontinuity between negative and possitive numbers.
        # Depending on whether the sweep increases or decreases the register, and on whether it crosses the
        # discontinuity or not, there are 4 scenarios:

        if self._abs_step > 0:  # increasing
            if (self._abs_start < 0 and self._abs_stop < 0) or (
                self._abs_stop > 0 and self._abs_start >= 0
            ):  # no crossing 0
                footer_block.append(
                    f"jlt {self.register}, {self._con_stop}, @loop_{self.register}",
                    comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                )
            elif self._abs_start < 0 and self._abs_stop >= 0:  # crossing 0
                # wait until the register crosses 0 to possitive values
                footer_block.append(
                    f"jge {self.register}, {2**31}, @loop_{self.register}",
                )
                # loop if the register is less than the stop value
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
            ):  # no crossing 0
                footer_block.append(
                    f"jge {self.register}, {self._con_stop + 1}, @loop_{self.register}",
                    comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                )
            elif self._abs_start >= 0 and self._abs_stop < 0:  # crossing 0
                if self._con_stop + 1 != 2**32:
                    # wait until the register crosses 0 to negative values
                    footer_block.append(
                        f"jlt {self.register}, {2**31}, @loop_{self.register}",
                    )
                    # loop if the register is greater than the stop value
                    footer_block.append(
                        f"jge {self.register}, {self._con_stop + 1}, @loop_{self.register}",
                        comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                    )
                else:  # special case when stopping at -1
                    footer_block.append(
                        f"jlt {self.register}, {2**31}, @loop_{self.register}",
                        comment=f"{self.register.name} loop, stop: {round(self._abs_stop, 6):_}",
                    )
            else:
                raise ValueError(
                    f"incorrect values for abs_start: {self._abs_start}, abs_stop: {self._abs_stop}, abs_step: {self._abs_step}"
                )

        return header_block + body_block + footer_block
