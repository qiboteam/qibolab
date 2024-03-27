import os
import pathlib

import numpy as np

from qibolab.pulses import (
    Drag,
    ECap,
    Gaussian,
    GaussianSquare,
    Iir,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
    Snz,
    plot,
)
from qibolab.pulses.modulation import modulate

HERE = pathlib.Path(__file__).parent
SAMPLING_RATE = 1


def test_plot_functions():
    p0 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=0,
        envelope=Rectangular(),
        relative_phase=0,
        type=PulseType.FLUX,
        qubit=0,
    )
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        relative_phase=0,
        type=PulseType.DRIVE,
        qubit=2,
    )
    p2 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Drag(rel_sigma=0.2, beta=2),
        relative_phase=0,
        type=PulseType.DRIVE,
        qubit=200,
    )
    p3 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Iir(a=np.array([-0.5, 2]), b=np.array([1]), target=Rectangular()),
        channel="0",
        qubit=200,
    )
    p4 = Pulse.flux(
        duration=40, amplitude=0.9, envelope=Snz(t_idling=10), channel="0", qubit=200
    )
    p5 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=400e6,
        envelope=ECap(alpha=2),
        relative_phase=0,
        type=PulseType.DRIVE,
    )
    p6 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=GaussianSquare(rel_sigma=0.2, width=0.9),
        relative_phase=0,
        type=PulseType.DRIVE,
        qubit=2,
    )
    ps = PulseSequence([p0, p1, p2, p3, p4, p5, p6])
    envelope = p0.envelopes(SAMPLING_RATE)
    wf = modulate(np.array(envelope), 0.0, rate=SAMPLING_RATE)

    plot_file = HERE / "test_plot.png"

    plot.waveform(wf, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.pulse(p0, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.sequence(ps, plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)
