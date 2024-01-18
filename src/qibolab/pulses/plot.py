"""Plotting tools for pulses and related entities."""
import matplotlib.pyplot as plt
import numpy as np

from .pulse import Pulse
from .shape import SAMPLING_RATE
from .waveform import Waveform


def waveform(wf: Waveform, filename=None):
    """Plot the waveform.

    Args:
        filename (str): a file path. If provided the plot is save to a file.
    """
    plt.figure(figsize=(14, 5), dpi=200)
    plt.plot(wf.data, c="C0", linestyle="dashed")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def pulse(pulse_: Pulse, filename=None, sampling_rate=SAMPLING_RATE):
    """Plot the pulse envelope and modulated waveforms.

    Args:
        filename (str): a file path. If provided the plot is save to a file.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    waveform_i = pulse_.shape.envelope_waveform_i(sampling_rate)
    waveform_q = pulse_.shape.envelope_waveform_q(sampling_rate)

    num_samples = len(waveform_i)
    time = pulse_.start + np.arange(num_samples) / sampling_rate
    _ = plt.figure(figsize=(14, 5), dpi=200)
    gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=np.array([2, 1]))
    ax1 = plt.subplot(gs[0])
    ax1.plot(
        time,
        waveform_i.data,
        label="envelope i",
        c="C0",
        linestyle="dashed",
    )
    ax1.plot(
        time,
        waveform_q.data,
        label="envelope q",
        c="C1",
        linestyle="dashed",
    )
    ax1.plot(
        time,
        pulse_.shape.modulated_waveform_i(sampling_rate).data,
        label="modulated i",
        c="C0",
    )
    ax1.plot(
        time,
        pulse_.shape.modulated_waveform_q(sampling_rate).data,
        label="modulated q",
        c="C1",
    )
    ax1.plot(time, -waveform_i.data, c="silver", linestyle="dashed")
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Amplitude")

    ax1.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
    start = float(pulse_.start)
    finish = float(pulse._finish) if pulse._finish is not None else 0.0
    ax1.axis((start, finish, -1.0, 1.0))
    ax1.legend()

    modulated_i = pulse_.shape.modulated_waveform_i(sampling_rate).data
    modulated_q = pulse_.shape.modulated_waveform_q(sampling_rate).data
    ax2 = plt.subplot(gs[1])
    ax2.plot(
        modulated_i,
        modulated_q,
        label="modulated",
        c="C3",
    )
    ax2.plot(
        waveform_i.data,
        waveform_q.data,
        label="envelope",
        c="C2",
    )
    ax2.plot(
        modulated_i[0],
        modulated_q[0],
        marker="o",
        markersize=5,
        label="start",
        c="lightcoral",
    )
    ax2.plot(
        modulated_i[-1],
        modulated_q[-1],
        marker="o",
        markersize=5,
        label="finish",
        c="darkred",
    )

    ax2.plot(
        np.cos(time * 2 * np.pi / pulse_.duration),
        np.sin(time * 2 * np.pi / pulse_.duration),
        c="silver",
        linestyle="dashed",
    )

    ax2.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
    ax2.legend()
    # ax2.axis([ -1, 1, -1, 1])
    ax2.axis("equal")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
