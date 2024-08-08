"""Plotting tools for pulses and related entities."""

from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .envelope import Waveform
from .modulation import modulate
from .pulse import Delay, Pulse, VirtualZ
from .sequence import PulseSequence

SAMPLING_RATE = 1
"""Default sampling rate in gigasamples per second (GSps).

Used for generating waveform envelopes if the instruments do not provide
a different value.
"""


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


def pulse(pulse_: Pulse, freq: Optional[float] = None, filename: Optional[str] = None):
    """Plot the pulse envelope and modulated waveforms.

    Args:
        freq: Carrier frequency used to plot modulated waveform. None if modulated plot is not needed.
        filename (str): a file path. If provided the plot is save to a file.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    waveform_i = pulse_.i(SAMPLING_RATE)
    waveform_q = pulse_.q(SAMPLING_RATE)

    num_samples = len(waveform_i)
    time = np.arange(num_samples) / SAMPLING_RATE
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

    envelope = pulse_.envelopes(SAMPLING_RATE)
    modulated = (
        modulate(np.array(envelope), freq, rate=SAMPLING_RATE)
        if freq is not None
        else None
    )

    if modulated is not None:
        ax1.plot(time, modulated[0], label="modulated i", c="C0")
        ax1.plot(time, modulated[1], label="modulated q", c="C1")

    ax1.plot(time, -waveform_i, c="silver", linestyle="dashed")
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Amplitude")

    ax1.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
    start = 0
    finish = float(pulse_.duration)
    ax1.axis((start, finish, -1.0, 1.0))
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    ax2.plot(waveform_i, waveform_q, label="envelope", c="C2")
    if modulated is not None:
        ax2.plot(modulated[0], modulated[1], label="modulated", c="C3")
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


def sequence(ps: PulseSequence, freq: dict[str, float], filename=None):
    """Plot the sequence of pulses.

    Args:
        freq: frequency per channel, used to plot modulated waveforms of corresponding pulses. If a channel is missing from this dict,
         only /un-modulated/ waveforms are plotted for that channel.
        filename (str): a file path. If provided the plot is save to a file.
    """
    if len(ps) > 0:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        num_pulses = len(ps)
        _ = plt.figure(figsize=(14, 2 * num_pulses), dpi=200)
        gs = gridspec.GridSpec(ncols=1, nrows=num_pulses)
        vertical_lines = []
        starts = defaultdict(float)
        for ch, pulse in ps:
            if not isinstance(pulse, Delay):
                vertical_lines.append(starts[ch])
                vertical_lines.append(starts[ch] + pulse.duration)
            starts[ch] += pulse.duration

        n = -1
        for ch in ps.channels:
            n += 1
            ax = plt.subplot(gs[n])
            ax.axis((0.0, ps.duration, -1.0, 1.0))
            start = 0
            for pulse in ps.channel(ch):
                if isinstance(pulse, (Delay, VirtualZ)):
                    start += pulse.duration
                    continue

                envelope = pulse.envelopes(SAMPLING_RATE)
                num_samples = envelope[0].size
                time = start + np.arange(num_samples) / SAMPLING_RATE
                if ch in freq:
                    modulated = modulate(
                        np.array(envelope), freq[ch], rate=SAMPLING_RATE
                    )
                    ax.plot(time, modulated[1], c="lightgrey")
                    ax.plot(time, modulated[0], c=f"C{str(n)}")
                ax.plot(time, pulse.i(SAMPLING_RATE), c=f"C{str(n)}")
                ax.plot(time, -pulse.i(SAMPLING_RATE), c=f"C{str(n)}")
                # TODO: if they overlap use different shades
                ax.axhline(0, c="dimgrey")
                ax.set_ylabel(f"channel {ch}")
                for vl in vertical_lines:
                    ax.axvline(vl, c="slategrey", linestyle="--")
                ax.axis((0, ps.duration, -1, 1))
                ax.grid(
                    visible=True,
                    which="both",
                    axis="both",
                    color="#CCCCCC",
                    linestyle="-",
                )
                start += pulse.duration

        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()
