import numpy as np
import pytest

from qibolab._core.instruments.qblox.sequence.asm import (
    MAX_PARAM,
    _convert_amplitude,
    _convert_frequency,
    _convert_offset,
    _convert_phase,
    _validate_sweeper_value,
    convert,
)
from qibolab._core.sweeper import Parameter


@pytest.mark.parametrize(
    "freq,expected", [(0.0, 0.0), (-100e6, -400e6), (-250e6, -1e9), (499e6, 499e6 * 4)]
)
def test_convert_frequency_valid(freq, expected):
    assert _convert_frequency(freq) == expected


@pytest.mark.parametrize("freq", [500e6, -500e6, -500.1e6, 6e9, -6e9])
def test_convert_frequency_invalid(freq):
    max_ = 500e6
    with pytest.raises(
        ValueError,
        match=(
            f"IF frequency must be a float between -{max_} and {max_}. Received: {freq}"
        ),
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
        ValueError,
        match=f"Offset must be a float between -1.0 and 1.0. Received: {offset}",
    ):
        _convert_offset(offset)


def test_convert_frequency():
    assert convert(100e6, Parameter.frequency) == 400e6
    assert convert(-100e6, Parameter.frequency) == (-400e6) % (2**32)
    with pytest.raises(ValueError, match="IF frequency must be a float between"):
        convert(6e9, Parameter.frequency)


def test_convert_offset():
    assert convert(0.5, Parameter.offset) == np.floor(0.5 * MAX_PARAM[Parameter.offset])
    assert convert(-0.5, Parameter.offset) == np.floor(
        -0.5 * MAX_PARAM[Parameter.offset]
    ) % (2**32)
    with pytest.raises(ValueError, match="Offset must be a float between"):
        convert(1.5, Parameter.offset)


@pytest.mark.parametrize("amplitude", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_convert_amplitude(amplitude):
    expected = amplitude * MAX_PARAM[Parameter.amplitude]
    assert _convert_amplitude(amplitude) == expected
    assert convert(amplitude, Parameter.amplitude) == expected


@pytest.mark.parametrize(
    "phase,turns",
    [(0.0, 0.0), (np.pi, 0.5), (-np.pi, 0.5), (2 * np.pi, 0.0), (3 * np.pi, 0.5)],
)
@pytest.mark.parametrize("kind", [Parameter.phase, Parameter.relative_phase])
def test_convert_phase(phase, turns, kind):
    expected = turns * MAX_PARAM[Parameter.phase]
    assert np.isclose(_convert_phase(phase), expected)
    assert np.isclose(convert(phase, kind), expected)


@pytest.mark.parametrize(
    "kind,converter,message",
    [
        (Parameter.amplitude, _convert_amplitude, "Amplitude must be a float between"),
        (Parameter.phase, _convert_phase, "Phase must be finite"),
        (Parameter.relative_phase, _convert_phase, "Phase must be finite"),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_convert_nonfinite(value, kind, converter, message):
    for validate in (
        converter,
        lambda v: convert(v, kind),
        lambda v: _validate_sweeper_value(v, kind),
    ):
        with pytest.raises(ValueError, match=message):
            validate(value)


@pytest.mark.parametrize("amplitude", [-1.5, 1.5])
def test_convert_amplitude_invalid(amplitude):
    for validate in (
        _convert_amplitude,
        lambda v: convert(v, Parameter.amplitude),
        lambda v: _validate_sweeper_value(v, Parameter.amplitude),
    ):
        with pytest.raises(ValueError, match="Amplitude must be a float between"):
            validate(amplitude)


def test_convert_duration():
    assert convert(100.0, Parameter.duration) == 100.0


def test_convert_unsupported():
    with pytest.raises(ValueError, match="Unsupported sweeper: duration_interpolated"):
        convert(1.0, Parameter.duration_interpolated)
