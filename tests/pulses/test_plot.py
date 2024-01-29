import os
import pathlib

from qibolab.pulses import (
    IIR,
    SNZ,
    Drag,
    Gaussian,
    GaussianSquare,
    Pulse,
    PulseSequence,
    PulseType,
    Rectangular,
    eCap,
    plot,
)

HERE = pathlib.Path(__file__).parent


def test_plot_functions():
    p0 = Pulse(0, 40, 0.9, 0, 0, Rectangular(), 0, PulseType.FLUX, 0)
    p1 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(5), 0, PulseType.DRIVE, 2)
    p2 = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 200)
    p3 = Pulse.flux(
        0, 40, 0.9, IIR([-0.5, 2], [1], Rectangular()), channel=0, qubit=200
    )
    p4 = Pulse.flux(0, 40, 0.9, SNZ(t_idling=10), channel=0, qubit=200)
    p5 = Pulse(0, 40, 0.9, 400e6, 0, eCap(alpha=2), 0, PulseType.DRIVE)
    p6 = Pulse(0, 40, 0.9, 50e6, 0, GaussianSquare(5, 0.9), 0, PulseType.DRIVE, 2)
    ps = PulseSequence([p0, p1, p2, p3, p4, p5, p6])
    wf = p0.modulated_waveform_i(0)

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
