import numpy as np
END_OF_LINE = '\n'

class Program():
    MAX_REGISTERS = 64

    def next_register(self):
        self._next_register_number += 1
        if self._next_register_number >= Program.MAX_REGISTERS:
            raise RuntimeError("There are no more registers available.")
        return self._next_register_number

    def __init__(self):
        self._blocks:list = []
        self._next_register_number:int=-1
    
    def add_blocks(self, *blocks):
        for block in blocks:
            self._blocks.append(block)

    def __repr__(self) -> str:
        block_str:str = ""
        for block in self._blocks:
            block_str += repr(block) + END_OF_LINE
        return block_str


class Block():
    GLOBAL_INDENTATION_LEVEL = 3
    SPACES_PER_LEVEL = 4
    SPACES_BEFORE_COMMENT = 4

    def __init__(self, name = ""):
        self.name = name
        self.lines:list = []
        self._indentation = 0

    def _indentation_string(self, level):
        return ' ' * Block.SPACES_PER_LEVEL * level

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

    def append(self, line, comment = "", level=0):
        self.lines = self.lines + [(line, comment, self._indentation + level)]
    
    def prepend(self, line, comment = "", level=0):
        self.lines = [(line, comment, self._indentation + level)] + self.lines

    def append_spacer(self):
        self.lines = self.lines + [("", "", self._indentation)]

    def __repr__(self) -> str:

        def comment_col(line, level):
            col = Block.SPACES_PER_LEVEL * (level + Block.GLOBAL_INDENTATION_LEVEL)
            col += len(line) 
            return col

        max_col: int = 0
        for (line, comment, level) in self.lines:
            if comment:
                max_col = max(max_col, comment_col(line, level))

        block_str:str = ""
        if self.name:
            block_str += self._indentation_string(self._indentation + Block.GLOBAL_INDENTATION_LEVEL) + "# " + self.name + END_OF_LINE

        for (line, comment, level) in self.lines:
            block_str += self._indentation_string(level + Block.GLOBAL_INDENTATION_LEVEL) + line
            if comment:
                block_str += ' ' * (max_col - comment_col(line, level) + Block.SPACES_BEFORE_COMMENT) + ' # ' + comment
            block_str +=  END_OF_LINE

        return block_str

 
    def __add__(self, other):
        if isinstance(other, Block):
            block = Block()
            for (line, comment, level) in self.lines:
                block.append(line, comment, level)
            for (line, comment, level) in other.lines:
                block.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return block

    def __radd__(self, other):
        if isinstance(other, Block):
            block = Block()
            for (line, comment, level) in other.lines:
                block.append(line, comment, level)
            for (line, comment, level) in self.lines:
                block.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return block

    def __iadd__(self, other):
        if isinstance(other, Block):
            for (line, comment, level) in other.lines:
                self.append(line, comment, level)
        else:
            raise TypeError(f"Expected Block, got {type(other).__name__}")
        return self


class Register():
    def __init__(self, program:Program, name:str=""): 
        self._number = program.next_register()
        self._name = name
    
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

    elif wait_time >= 4 and wait_time < 2**16: # 65536ns
        n_loops = 0
        loop_wait = 0
        wait = wait_time
    elif wait_time >= 2**16 and wait_time < 2**32: # 4.29s
        loop_wait = 2**16 -4
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
        wait = int(np.ceil(wait/4)) * 4

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


def sweeper_block(begin: int, end: int, step: int, register: Register, block, update_parameter_block = None):
    # validate values 
    if not (isinstance(begin, int) and isinstance(end, int) and isinstance(step, int)):
        raise ValueError("begin, end and step must be int")
    if end != begin and step == 0:
        raise ValueError("step must not be 0")
    if (end > begin and not step > 0) or (end < begin and not step < 0):
        raise ValueError("invalid step")

    header_block = Block()
    # same behaviour as range() includes the first but never the last
    if begin >= 0:
        header_block.append(f"move {begin}, {register}", comment=register.name + " loop")
        header_block.append("nop")
    else:
        header_block.append(f"move 0, {register}", comment=register.name + " loop")
        header_block.append("nop")
        header_block.append(f"sub {register}, {abs(begin)}, {register}")
        header_block.append("nop")
        
    header_block.append(f"loop_{register}:")
    header_block.append("")

    body_block = Block()
    body_block.indentation = 1
    body_block += block

    footer_block = Block()
    footer_block.append("")
    if end > begin:
        footer_block.append(f"add {register}, {step}, {register}")
        footer_block.append("nop")
        if update_parameter_block:
            footer_block += update_parameter_block
        footer_block.append(f"jlt {register}, {end}, @loop_{register}", comment=register.name + " loop")
    elif end < begin:
        footer_block.append(f"sub {register}, {abs(step)}, {register}")
        footer_block.append("nop")
        if update_parameter_block:
            footer_block += update_parameter_block
        footer_block.append(f"jge {register}, {end + 1}, @loop_{register}", comment=register.name + " loop")

    return header_block + body_block + footer_block

def loop_block(begin: int, end: int, step: int, register: Register, block):
    return sweeper_block(begin, end, step, register, block, None)