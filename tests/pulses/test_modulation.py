import numpy as np

from qibolab._core.pulses import Gaussian, IqWaveform, Rectangular
from qibolab._core.pulses.modulation import demodulate, modulate


def test_modulation():
    amplitude = 0.9
    renvs: IqWaveform = Rectangular().envelopes(30) * amplitude
    # fmt: off
    np.testing.assert_allclose(modulate(renvs, 0.04, rate=1),
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

    genvs: IqWaveform = Gaussian(rel_sigma=0.5).envelopes(20)
    # fmt: off
    np.testing.assert_allclose(modulate(genvs, 0.3,rate=1),
         np.array([[ 4.50307953e-01, -1.52257426e-01, -4.31814602e-01,
                     4.63124693e-01,  1.87836646e-01, -6.39017403e-01,
                     2.05526028e-01,  5.54460924e-01, -5.65661777e-01,
                    -2.18235048e-01,  7.06223450e-01, -2.16063573e-01,
                    -5.54460924e-01,  5.38074127e-01,  1.97467237e-01,
                    -6.07852156e-01,  1.76897892e-01,  4.31814602e-01,
                    -3.98615117e-01, -1.39152810e-01],
                   [ 0.00000000e+00,  4.68600175e-01, -3.13731672e-01,
                    -3.36479785e-01,  5.78101754e-01,  2.34771185e-16,
                    -6.32544073e-01,  4.02839441e-01,  4.10977338e-01,
                    -6.71658414e-01, -5.18924572e-16,  6.64975301e-01,
                    -4.02839441e-01, -3.90933736e-01,  6.07741665e-01,
                     6.69963778e-16, -5.44435729e-01,  3.13731672e-01,
                     2.89610835e-01, -4.28268313e-01]])
    )
    # fmt: on


def test_demodulation():
    signal = np.ones((2, 100))
    freq = 0.15
    rate = 1
    mod = modulate(signal, freq, rate)

    demod = demodulate(mod, freq, rate)
    np.testing.assert_allclose(demod, signal)

    mod1 = modulate(demod, freq * 3.0, rate=3.0)
    np.testing.assert_allclose(mod1, mod)

    mod2 = modulate(signal, freq, rate, phase=2 * np.pi)
    np.testing.assert_allclose(mod2, mod)

    demod1 = demodulate(mod + np.ones_like(mod), freq, rate)
    np.testing.assert_allclose(demod1, demod)
