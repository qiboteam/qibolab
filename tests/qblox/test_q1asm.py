from pathlib import Path

import pytest

from qibolab._core.instruments.qblox.q1asm import Program, parse

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


def test_serialization(asm):
    prog = parse(asm)
    assert prog == Program.model_validate(prog.model_dump())


def test_json_serialization(asm):
    prog = parse(asm)
    assert prog == Program.model_validate_json(prog.model_dump_json())
