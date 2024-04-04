from pathlib import Path

from qibolab.instruments.neoqblox.parse import parse

PROGRAMS = Path(__file__).parent / "q1asm"


def test_q1asm():
    for program in PROGRAMS.glob("*.q1asm"):
        parse(program.read_text())
