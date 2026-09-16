import numpy as np
import pytest

from qibolab._core.instruments.qblox.sequence.asm import (
    MAX_PARAM,
    _convert_frequency,
    _convert_offset,
    convert,
)
from qibolab._core.sweeper import Parameter


@pytest.mark.parametrize(
    "freq,expected", [(0.0, 0.0), (-100e6, -400e6), (-250e6, -1e9), (499e6, 499e6 * 4)]
)
def test_convert_frequency_valid(freq, expected):
    assert _convert_frequency(freq) == expected


@pytest.mark.parametrize("freq", [500e6, -500.1e6, 6e9, -6e9])
def test_convert_frequency_invalid(freq):
    with pytest.raises(
        ValueError,
        match=f"Frequency must be a float between -500e6 and 500e6. Received: {freq}",
    ):
        _convert_frequency(freq)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0.0, 0.0),
        (0.5, np.floor(0.5 * MAX_PARAM[Parameter.offset])),
        (-0.5, np.floor(-0.5 * MAX_PARAM[Parameter.offset])),
    ],
)
def test_convert_offset_valid(offset, expected):
    assert _convert_offset(offset) == expected


@pytest.mark.parametrize("offset", [1.0, -1.0, 1.5, -1.5])
def test_convert_offset_invalid(offset):
    with pytest.raises(
        ValueError, match=f"Offset must be a float between -1 and 1. Received: {offset}"
    ):
        _convert_offset(offset)


def test_convert_frequency():
    assert convert(100e6, Parameter.frequency) == 400e6
    assert convert(-100e6, Parameter.frequency) == (-400e6) % (2**32)
    with pytest.raises(ValueError, match="Frequency must be a float between"):
        convert(6e9, Parameter.frequency)


def test_convert_offset():
    assert convert(0.5, Parameter.offset) == np.floor(0.5 * MAX_PARAM[Parameter.offset])
    assert convert(-0.5, Parameter.offset) == np.floor(
        -0.5 * MAX_PARAM[Parameter.offset]
    ) % (2**32)
    with pytest.raises(ValueError, match="Offset must be a float between"):
        convert(1.5, Parameter.offset)


def test_convert_amplitude():
    assert convert(0.5, Parameter.amplitude) == 0.5 * MAX_PARAM[Parameter.amplitude]


def test_convert_phase():
    assert convert(np.pi, Parameter.phase) == 0.5 * MAX_PARAM[Parameter.phase]
    assert convert(np.pi, Parameter.relative_phase) == 0.5 * MAX_PARAM[Parameter.phase]
    assert np.isclose(
        convert(3 * np.pi, Parameter.phase),
        0.5 * MAX_PARAM[Parameter.phase],
    )


def test_convert_duration():
    assert convert(100.0, Parameter.duration) == 100.0


def test_convert_unsupported():
    with pytest.raises(ValueError, match="Unsupported sweeper: duration_interpolated"):
        convert(1.0, Parameter.duration_interpolated)
