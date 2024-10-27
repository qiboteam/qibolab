from pathlib import Path

import pytest

from qibolab._core.instruments.qblox.ast_ import Program
from qibolab._core.instruments.qblox.parse import parse

PROGRAMS = Path(__file__).parent / "q1asm"


@pytest.fixture(params=PROGRAMS.glob("*.q1asm"), ids=lambda p: p.stem)
def asm(request):
    return request.param.read_text()


def test_load(asm):
    prog = parse(asm)
    assert isinstance(prog, Program)


def test_dump(asm):
    prog = parse(asm)
    assert isinstance(prog.asm(), str)


def test_roundtrip(asm):
    prog = parse(asm)
    assert prog == parse(prog.asm())
