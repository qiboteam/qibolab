"""Tests ``components/configs.py``."""

import pytest

pytest.importorskip("qm")

from qibolab._core.components.filters import (
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.instruments.qm.components.configs import OpxOutputConfig

FIR_COEFFICIENTS = [0.97, 0.36, -0.18, -0.10]
EXPONENTIAL = ExponentialFilter(amplitude=-0.0089, tau=100.0)


@pytest.mark.parametrize("cluster", ["opx1000", "LF", "MW"])
def test_opx1000_feedforward_excludes_exponential(cluster):
    """The FIR ``feedforward`` should only contain genuine FIR taps.

    ``ExponentialFilter``s are applied through the dedicated
    ``exponential`` IIR stage on these clusters, and must not also be
    folded into ``feedforward`` (which would double-apply them).

    Cf. https://github.com/qiboteam/qibolab/issues/1547
    """
    config = OpxOutputConfig(
        offset=0.0,
        filters=[
            FiniteImpulseResponseFilter(coefficients=FIR_COEFFICIENTS),
            EXPONENTIAL,
        ],
    )

    filt = config.filter(cluster)

    assert filt["feedforward"] == FIR_COEFFICIENTS
    assert filt["exponential"] == [(EXPONENTIAL.amplitude, EXPONENTIAL.tau)]


def test_opx1000_feedforward_empty_with_only_exponential():
    """No FIR filter configured means no FIR stage at all, not an
    approximation of the exponential correction."""
    config = OpxOutputConfig(offset=0.0, filters=[EXPONENTIAL])

    filt = config.filter("LF")

    assert filt["feedforward"] == []
    assert filt["exponential"] == [(EXPONENTIAL.amplitude, EXPONENTIAL.tau)]


def test_opx1000_feedforward_no_filters():
    config = OpxOutputConfig(offset=0.0, filters=[])

    filt = config.filter("LF")

    assert filt == {"feedforward": [], "exponential": []}


def test_opx1_still_folds_exponential_into_feedforward():
    """On the legacy OPX+ there is no dedicated ``exponential`` stage, so
    folding the exponential's own FIR/IIR approximation into
    ``feedforward``/``feedback`` remains the correct (and only) way to
    apply it; this branch must be unaffected by the OPX1000 fix."""
    config = OpxOutputConfig(
        offset=0.0,
        filters=[
            FiniteImpulseResponseFilter(coefficients=FIR_COEFFICIENTS),
            EXPONENTIAL,
        ],
    )

    filt = config.filter("opx1")

    assert filt["feedforward"] != FIR_COEFFICIENTS
    assert len(filt["feedforward"]) == len(FIR_COEFFICIENTS) + 1
    assert filt["feedback"] == [-EXPONENTIAL.feedback[1]]


def test_filter_unsupported_cluster_raises():
    config = OpxOutputConfig(offset=0.0)

    with pytest.raises(NotImplementedError):
        config.filter("unknown")
