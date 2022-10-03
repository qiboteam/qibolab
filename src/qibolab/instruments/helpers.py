# -*- coding: utf-8 -*-
import numpy as np


def if_allocation(freq, bandwidth=600e6, peak_width=20e6):
    """
    Sets the LO to be the center frequency + peak_width to avoid spurs' overlap

    Param:
    freq: the list of IF frequencies for which to choose an LO
    bandwidth: bandwidth of the instrument
    peak_width: typical peaks width

    Return:
    LO_optimal: optimal lo frequency
    if: IF frequencies to setup
    """

    dx = bandwidth - (max(freq) - min(freq))
    if dx <= 0:
        raise ValueError("Bandwitch too small for resonator's spacing")
    elif dx < peak_width:
        peak_width = dx

    if len(freq) > 1:
        lo = peak_width + (min(freq) + max(freq)) / 2
    else:
        lo = freq - bandwidth / 2
    return lo


def if_allocation_vain(freq, bandwidth=600e6, weights=[1, 1, 1, 1]):
    """
    Function suppose to set the center frequency (LO) to avoid that spurs overlap the desired frequencies. Few problems:
    - scipy.minimize not working well, so it is simply iterating through all values
    - weights are too hard to choose, so a different weighting method should be used
    - for most weights, the solution is to set the LO to be furthest to most peaks, this would result in spurs greatly spaced
    which might work well

    Param:
    bandwidth: bandwidth of the instrument
    weights: factor to which to value important of a peak [image_spur, image, lo_leake, rf_spur]

    Return:
    LO_optimal: optimal lo frequency
    """
    scale = 1 / max(freq)
    bandwidth = np.array(bandwidth) * scale
    freq = np.array(freq) * scale

    rn = list(range(len(freq)))

    dx = bandwidth - (max(freq) - min(freq))
    if dx <= 0:
        raise ValueError("Bandwitch too small for resonator's spacing")
    freq_max = max(freq) - (bandwidth / 2 - dx)
    freq_min = min(freq) + (bandwidth / 2 - dx)

    def cost(LO):
        LO = LO[0]
        freqs = np.zeros((len(rn), 4))

        for i in rn:
            freqs[i, :] = np.array([-2 * (freq[i] - LO), -(freq[i] - LO), 0, 2 * (freq[i] - LO)]) + np.array(LO)
        c = 0
        for i in rn:
            for j in rn:
                if i != j:
                    c += np.sum((1) / (1 + weights * np.abs(freq[i] - freqs[j, :])))
        return c

    # result = minimize(lambda x: cost(x), x0 = (min(freq)+max(freq))/2, method="Nelder-Mead", bounds=[(freq_min, freq_max)])
    x = np.arange(freq_min, freq_max, 100e3 * scale)
    y = []
    for f in x:
        y += [cost([f])]

    LO_optimal = x[np.argmin(y)] / scale

    return LO_optimal
