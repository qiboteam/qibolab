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
    )
    p1 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Gaussian(rel_sigma=0.2),
        relative_phase=0,
        type=PulseType.DRIVE,
    )
    p2 = Pulse(
        duration=40,
        amplitude=0.9,
        frequency=50e6,
        envelope=Drag(rel_sigma=0.2, beta=2),
        relative_phase=0,
        type=PulseType.DRIVE,
    )
    p3 = Pulse.flux(
        duration=40,
        amplitude=0.9,
        envelope=Iir(a=np.array([-0.5, 2]), b=np.array([1]), target=Rectangular()),
    )
    p4 = Pulse.flux(duration=40, amplitude=0.9, envelope=Snz(t_idling=10))
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
    )
    ps = PulseSequence()
    ps.update(
        {
            "q0/flux": [p0],
            "q2/drive": [p1, p6],
            "q200/drive": [p2],
            "q200/flux": [p3, p4],
            "q0/drive": [p5],
        }
    )
    envelope = p0.envelopes(SAMPLING_RATE)
    wf = modulate(np.array(envelope), 0.0, rate=SAMPLING_RATE)

    plot_file = HERE / "test_plot.png"

    plot.waveform(wf, filename=plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.pulse(p0, filename=plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.pulse(p0, freq=2e9, filename=plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)

    plot.sequence(ps, {"q200/drive": 3e9}, filename=plot_file)
    assert os.path.exists(plot_file)
    os.remove(plot_file)
