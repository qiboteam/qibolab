import numpy as np

END_OF_LINE = "\n"


class Program:
    MAX_REGISTERS = 64

    def next_register(self):
        self._next_register_number += 1
        if self._next_register_number >= Program.MAX_REGISTERS:
            raise RuntimeError("There are no more registers available.")
        return self._next_register_number

    def __init__(self):
        self._blocks: list = []
        self._next_register_number: int = -1

    def add_blocks(self, *blocks):
        for block in blocks:
            self._blocks.append(block)

    def __repr__(self) -> str:
        block_str: str = ""
        for block in self._blocks:
            block_str += repr(block) + END_OF_LINE
        return block_str


class Block:
    GLOBAL_INDENTATION_LEVEL = 3
    SPACES_PER_LEVEL = 4
    SPACES_BEFORE_COMMENT = 4

    def __init__(self, name=""):
        self.name = name
        self.lines: list = []
        self._indentation = 0

    def _indentation_string(self, level):
        return " " * Block.SPACES_PER_LEVEL * level

    @property
    def indentation(self):
        return self._indentation

    @indentation.setter
    def indentation(self, value):
        if not isinstance(value, int):
            raise TypeError(f"indentation type should be int, got {type(value).__name__}")

        diff = value - self._indentation
        self._indentation = value
        self.lines = [(line, comment, level + diff) for (line, comment, level) in self.lines]

    def append(self, line, comment="", level=0):
        self.lines = self.lines + [(line, comment, self._indentation + level)]

    def prepend(self, line, comment="", level=0):
        self.lines = [(line, comment, self._indentation + level)] + self.lines

    def append_spacer(self):
        self.lines = self.lines + [("", "", self._indentation)]

    def __repr__(self) -> str:
        def comment_col(line, level):
            col = Block.SPACES_PER_LEVEL * (level + Block.GLOBAL_INDENTATION_LEVEL)
            col += len(line)
            return col

        max_col: int = 0
        for line, comment, level in self.lines:
            if comment:
                max_col = max(max_col, comment_col(line, level))

        block_str: str = ""
        if self.name:
            block_str += (
                self._indentation_string(self._indentation + Block.GLOBAL_INDENTATION_LEVEL)
                + "# "
                + self.name
                + END_OF_LINE
            )

        for line, comment, level in self.lines:
            block_str += self._indentation_string(level + Block.GLOBAL_INDENTATION_LEVEL) + line
            if comment:
                block_str += " " * (max_col - comment_col(line, level) + Block.SPACES_BEFORE_COMMENT) + " # " + comment
            block_str += END_OF_LINE

        return block_str

    def __add__(self, other):
        if isinstance(other, Block):
            block = Block()
            for line, comment, level in self.lines:
                block.append(line, comment, level)
            for line, comment, level in other.lines:
                block.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return block

    def __radd__(self, other):
        if isinstance(other, Block):
            block = Block()
            for line, comment, level in other.lines:
                block.append(line, comment, level)
            for line, comment, level in self.lines:
                block.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return block

    def __iadd__(self, other):
        if isinstance(other, Block):
            for line, comment, level in other.lines:
                self.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return self


class Register:
    def __init__(self, program: Program, name: str = ""):
        self._number = program.next_register()
        self._name = name
        self._type = type

    def __repr__(self) -> str:
        return "R" + str(self._number)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


def wait_block(wait_time: int, register: Register, force_multiples_of_4: bool = False):
    n_loops: int
    loop_wait: int
    wait: int
    block = Block()

    # constrains
    # extra_wait and wait_loop_step need to be within (4,65535) (2**16 bits variable)
    # extra_wait and wait_loop_step need to be multiples of 4ns *

    if wait_time < 0:
        raise ValueError("wait_time must be possitive.")

    elif wait_time == 0:
        n_loops = 0
        loop_wait = 0
        wait = 0

    elif wait_time > 0 and wait_time < 4:
        # TODO: log("wait_time > 0 and wait_time < 4 is not supported by the instrument, wait_time changed to 4ns")
        n_loops = 0
        loop_wait = 0
        wait = 4

    elif wait_time >= 4 and wait_time < 2**16:  # 65536ns
        n_loops = 0
        loop_wait = 0
        wait = wait_time
    elif wait_time >= 2**16 and wait_time < 2**32:  # 4.29s
        loop_wait = 2**16 - 4
        n_loops = wait_time // loop_wait
        wait = wait_time % loop_wait

        while loop_wait >= 4 and (wait > 0 and wait < 4):
            loop_wait -= 4
            n_loops = wait_time // loop_wait
            wait = wait_time % loop_wait

        if loop_wait < 4 or (wait > 0 and wait < 4):
            raise ValueError(f"Unable to decompose {wait_time} into valid (n_loops * loop_wait + wait)")
    else:
        raise ValueError("wait_time > 65535**2 is not supported yet.")

    if force_multiples_of_4:
        wait = int(np.ceil(wait / 4)) * 4

    wait_time = n_loops * loop_wait + wait

    if n_loops > 0:
        block.append(f"# wait {wait_time} ns")
        block.append(f"move {n_loops}, {register}")
        block.append(f"wait_loop_{register}:")
        block.append(f"wait {loop_wait}", level=1)
        block.append(f"loop {register}, @wait_loop_{register}", level=1)
    if wait > 0:
        block.append(f"wait {wait}")

    return block


def loop_block(start: int, stop: int, step: int, register: Register, block):  # validate values
    if not (isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)):
        raise ValueError("begin, end and step must be int")
    if stop != start and step == 0:
        raise ValueError("step must not be 0")
    if (stop > start and not step > 0) or (stop < start and not step < 0):
        raise ValueError("invalid step")
    if start < 0 or stop < 0:
        raise ValueError("sweep possitive values only")

    header_block = Block()
    # same behaviour as range() includes the first but never the last
    header_block.append(f"move {start}, {register}", comment=register.name + " loop")
    header_block.append("nop")
    header_block.append(f"loop_{register}:")
    header_block.append_spacer()

    body_block = Block()
    body_block.indentation = 1
    body_block += block

    footer_block = Block()
    footer_block.append_spacer()
    if stop > start:
        footer_block.append(f"add {register}, {step}, {register}")
        footer_block.append("nop")
        footer_block.append(f"jlt {register}, {stop}, @loop_{register}", comment=register.name + " loop")
    elif stop < start:
        footer_block.append(f"sub {register}, {abs(step)}, {register}")
        footer_block.append("nop")
        footer_block.append(f"jge {register}, {stop + 1}, @loop_{register}", comment=register.name + " loop")

    return header_block + body_block + footer_block


def convert_phase(phase_rad: float):
    phase_deg = (phase_rad * 360 / (2 * np.pi)) % 360
    return int(phase_deg / 360 * 1e9)
    """
    The phase is divided into 1e9 steps between 0° and 360°,
    expressed as an integer between 0 and 1e9 (e.g 45°=125e6).
    """


def convert_frequency(freq: float):
    if not (freq >= -500e6 and freq <= 500e6):
        raise ValueError("frequency must be a float between -500e6 and 500e6 Hz")
    return int(freq * 4) % 2**32  # two's complement of 18? TODO: confirm with qblox
    """
    The frequency is divided into 4e9 steps between -500 and 500 MHz and
    expressed as an integer between -2e9 and 2e9. (e.g. 1 MHz=4e6).
    """


def convert_gain(gain: float):
    if not (gain >= -1 and gain <= 1):
        raise ValueError("gain must be a float between -1 and 1")
    if gain == 1:
        return 2**15 - 1
    else:
        return int(np.floor(gain * 2**15)) % 2**32  # two's complement 32 bit number
    """ Both gain values are divided in 2**sample path width steps."""
    """ QCM DACs resolution 16bits, QRM DACs and ADCs 12 bit"""


def convert_offset(offset: float):
    scale_factor = 1.25 * np.sqrt(2)
    normalised_offset = offset / scale_factor

    if not (normalised_offset >= -1 and normalised_offset <= 1):
        raise ValueError(f"offset must be a float between {-scale_factor:.3f} and {scale_factor:.3f} V")
    if normalised_offset == 1:
        return 2**15 - 1
    else:
        return int(np.floor(normalised_offset * 2**15)) % 2**32  # two's complement 32 bit number? or 12 or 24?
    """ Both offset values are divided in 2**sample path width steps."""
    """ QCM DACs resolution 16bits, QRM DACs and ADCs 12 bit"""
    """ QCM 5Vpp, QRM 2Vpp"""


# https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/tutorials/nco.html#Fast-chirped-pulses-using-Q1ASM
"""
The sequencer program can fundamentally only support integer values.
However, the NCO has a frequency resolution of 0.25 Hz and supports 1e9 phase values.
Therefore, frequencies in the sequencer program must be given as an integer multiple of 1/4 Hz,
and phases as an integer multiple of 360/1e9 degrees.

Internally, the processor stores negative values using two’s complement.
This has some implications for our program:
- We cannot directly store a negative value in a register.
   Substracting a larger value from a smaller one works as expected though.
- Immediate values are handled by the compiler, i.e. set_freq-100 gives the expected result of -25 Hz.
- Comparisons (jlt, jge) with registers storing a negative value do not work as expected,
   as the smallest negative number is larger than the largest positive number.
   To keep the program general we should therefore use loop instead.
"""
