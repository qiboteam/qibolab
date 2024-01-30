"""Plotting tools for pulses and related entities."""

import matplotlib.pyplot as plt
import numpy as np

from .pulse import Pulse
from .sequence import PulseSequence
from .shape import SAMPLING_RATE, Waveform, modulate


def waveform(wf: Waveform, filename=None):
    """Plot the waveform.

    Args:
        filename (str): a file path. If provided the plot is save to a file.
    """
    plt.figure(figsize=(14, 5), dpi=200)
    plt.plot(wf, c="C0", linestyle="dashed")
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
        waveform_i,
        label="envelope i",
        c="C0",
        linestyle="dashed",
    )
    ax1.plot(
        time,
        waveform_q,
        label="envelope q",
        c="C1",
        linestyle="dashed",
    )

    envelope = pulse_.shape.envelope_waveforms(sampling_rate)
    modulated = modulate(np.array(envelope), pulse_.frequency)
    ax1.plot(time, modulated[0], label="modulated i", c="C0")
    ax1.plot(time, modulated[1], label="modulated q", c="C1")
    ax1.plot(time, -waveform_i, c="silver", linestyle="dashed")
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Amplitude")

    ax1.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
    start = float(pulse_.start)
    finish = float(pulse_.finish) if pulse_.finish is not None else 0.0
    ax1.axis((start, finish, -1.0, 1.0))
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    ax2.plot(modulated[0], modulated[1], label="modulated", c="C3")
    ax2.plot(waveform_i, waveform_q, label="envelope", c="C2")
    ax2.plot(
        modulated[0][0],
        modulated[1][0],
        marker="o",
        markersize=5,
        label="start",
        c="lightcoral",
    )
    ax2.plot(
        modulated[0][-1],
        modulated[1][-1],
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


def sequence(ps: PulseSequence, filename=None, sampling_rate=SAMPLING_RATE):
    """Plot the sequence of pulses.

    Args:
        filename (str): a file path. If provided the plot is save to a file.
    """
    if len(ps) > 0:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        _ = plt.figure(figsize=(14, 2 * len(ps)), dpi=200)
        gs = gridspec.GridSpec(ncols=1, nrows=len(ps))
        vertical_lines = []
        for pulse in ps:
            vertical_lines.append(pulse.start)
            vertical_lines.append(pulse.finish)

        n = -1
        for qubit in ps.qubits:
            qubit_pulses = ps.get_qubit_pulses(qubit)
            for channel in qubit_pulses.channels:
                n += 1
                channel_pulses = qubit_pulses.get_channel_pulses(channel)
                ax = plt.subplot(gs[n])
                ax.axis([0, ps.finish, -1, 1])
                for pulse in channel_pulses:
                    envelope = pulse.shape.envelope_waveforms(sampling_rate)
                    num_samples = envelope[0].size
                    time = pulse.start + np.arange(num_samples) / sampling_rate
                    modulated = modulate(np.array(envelope), pulse.frequency)
                    ax.plot(time, modulated[1], c="lightgrey")
                    ax.plot(time, modulated[0], c=f"C{str(n)}")
                    ax.plot(
                        time,
                        pulse.shape.envelope_waveform_i(sampling_rate),
                        c=f"C{str(n)}",
                    )
                    ax.plot(
                        time,
                        -pulse.shape.envelope_waveform_i(sampling_rate),
                        c=f"C{str(n)}",
                    )
                    # TODO: if they overlap use different shades
                    ax.axhline(0, c="dimgrey")
                    ax.set_ylabel(f"qubit {qubit} \n channel {channel}")
                    for vl in vertical_lines:
                        ax.axvline(vl, c="slategrey", linestyle="--")
                    ax.axis((0, ps.finish, -1, 1))
                    ax.grid(
                        visible=True,
                        which="both",
                        axis="both",
                        color="#CCCCCC",
                        linestyle="-",
                    )
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()
