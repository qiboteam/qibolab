import numpy as np

from qibolab.pulses import Envelopes, IqWaveform, Pulse, PulseType
from qibolab.pulses.modulation import demodulate, modulate

Rectangular = Envelopes.RECTANGULAR.value
Gaussian = Envelopes.GAUSSIAN.value


def test_modulation():
    rect = Pulse(
        start=0,
        duration=30,
        amplitude=0.9,
        frequency=20_000_000,
        relative_phase=0.0,
        shape=Rectangular(),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    renvs: IqWaveform = np.array(rect.shape.envelope_waveforms())
    # fmt: off
    np.testing.assert_allclose(modulate(renvs, 0.04),
         np.array([[ 6.36396103e-01,  6.16402549e-01,  5.57678156e-01,
                     4.63912794e-01,  3.40998084e-01,  1.96657211e-01,
                     3.99596419e-02, -1.19248738e-01, -2.70964282e-01,
                    -4.05654143e-01, -5.14855263e-01, -5.91706132e-01,
                    -6.31377930e-01, -6.31377930e-01, -5.91706132e-01,
                    -5.14855263e-01, -4.05654143e-01, -2.70964282e-01,
                    -1.19248738e-01,  3.99596419e-02,  1.96657211e-01,
                     3.40998084e-01,  4.63912794e-01,  5.57678156e-01,
                     6.16402549e-01,  6.36396103e-01,  6.16402549e-01,
                     5.57678156e-01,  4.63912794e-01,  3.40998084e-01],
                   [ 0.00000000e+00,  1.58265275e-01,  3.06586161e-01,
                     4.35643111e-01,  5.37327002e-01,  6.05248661e-01,
                     6.35140321e-01,  6.25123778e-01,  5.75828410e-01,
                     4.90351625e-01,  3.74064244e-01,  2.34273031e-01,
                     7.97615814e-02, -7.97615814e-02, -2.34273031e-01,
                    -3.74064244e-01, -4.90351625e-01, -5.75828410e-01,
                    -6.25123778e-01, -6.35140321e-01, -6.05248661e-01,
                    -5.37327002e-01, -4.35643111e-01, -3.06586161e-01,
                    -1.58265275e-01,  4.09361195e-16,  1.58265275e-01,
                     3.06586161e-01,  4.35643111e-01,  5.37327002e-01]])
    )
    # fmt: on

    gauss = Pulse(
        start=5,
        duration=20,
        amplitude=3.5,
        frequency=2_000_000,
        relative_phase=0.0,
        shape=Gaussian(0.5),
        channel=0,
        type=PulseType.READOUT,
        qubit=0,
    )
    genvs: IqWaveform = np.array(gauss.shape.envelope_waveforms())
    # fmt: off
    np.testing.assert_allclose(modulate(genvs, 0.3),
        np.array([[ 2.40604965e+00, -7.47704261e-01, -1.96732725e+00,
                    1.97595317e+00,  7.57582564e-01, -2.45926187e+00,
                    7.61855973e-01,  1.99830815e+00, -2.00080760e+00,
                   -7.64718297e-01,  2.47468039e+00, -7.64240497e-01,
                   -1.99830815e+00,  1.99456483e+00,  7.59953712e-01,
                   -2.45158868e+00,  7.54746949e-01,  1.96732725e+00,
                   -1.95751517e+00, -7.43510231e-01],
                  [ 0.00000000e+00,  2.30119709e+00, -1.42934692e+00,
                   -1.43561401e+00,  2.33159938e+00,  9.03518154e-16,
                   -2.34475159e+00,  1.45185586e+00,  1.45367181e+00,
                   -2.35356091e+00, -1.81836565e-15,  2.35209040e+00,
                   -1.45185586e+00, -1.44913618e+00,  2.33889703e+00,
                    2.70209720e-15, -2.32287226e+00,  1.42934692e+00,
                    1.42221802e+00, -2.28828920e+00]])
    )
    # fmt: on


def test_demodulation():
    signal = np.ones((2, 100))
    freq = 0.15
    mod = modulate(signal, freq)

    demod = demodulate(mod, freq)
    np.testing.assert_allclose(demod, signal)

    mod1 = modulate(demod, freq * 3.0, rate=3.0)
    np.testing.assert_allclose(mod1, mod)

    mod2 = modulate(signal, freq, phase=2 * np.pi)
    np.testing.assert_allclose(mod2, mod)

    demod1 = demodulate(mod + np.ones_like(mod), freq)
    np.testing.assert_allclose(demod1, demod)
