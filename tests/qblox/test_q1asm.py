from pathlib import Path

from qibolab.instruments.neoqblox.ast_ import Program
from qibolab.instruments.neoqblox.parse import parse

PROGRAMS = Path(__file__).parent / "q1asm"


def test_q1asm():
    for program in PROGRAMS.glob("*.q1asm"):
        prog = parse(program.read_text())
        assert isinstance(prog, Program)
