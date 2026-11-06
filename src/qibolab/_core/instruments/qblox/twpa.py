"""TWPA utility functions for Qblox instruments."""

import warnings

from qblox_instruments.qcodes_drivers.module import Module

__all__ = []


def max_output_attenuation(module: Module, output: int) -> int:
    """Maximum attenuation of a module output [dB].

    It depends on the module: 60 dB for the QRM-RF and the original QCM-RF, 30 dB for
    the QCM-RF II (cf. `QCM-RF II
    <https://docs.qblox.com/en/main/products/architecture/modules/qcm_rf.html>`_).

    It is retrieved as done by the Qblox driver itself, to bound the ``out<n>_att``
    parameters, since no public getter is exposed for QCM and QRM modules.
    """
    return int(getattr(module, f"_get_max_out_att_{output}")())


def twpa_attenuation_and_offset(
    power: float, max_attenuation: int
) -> tuple[int, float]:
    """Calculate hardware port attenuation (in dB) and fine-tuning offset factor.

    Follows the standard Qibolab convention, where the power is the opposite of the
    attenuation, relative to the full scale of the port (thus ``power <= 0``).
    The port attenuation is the largest even integer not exceeding the target one, since
    it is programmable in 2 dB steps (cf. `QCM-RF II
    <https://docs.qblox.com/en/main/products/architecture/modules/qcm_rf.html>`_), and
    the remainder is applied by the offset, ensuring ``offset <= 1``.

    The port attenuation is clamped to its available range, up to ``max_attenuation``
    (cf. :func:`max_output_attenuation`): beyond the maximum, the offset applies the
    whole remainder, while powers above the full scale are clamped to it, with a
    warning.

    Args:
        power: Power in dB, relative to the full scale (conventionally ``<= 0``).
        max_attenuation: Maximum attenuation of the port, in dB.

    Returns:
        tuple[int, float]: (attenuation_in_dB, fine_tuning_offset)
    """
    if power > 0:
        warnings.warn(
            f"Power {power} dB exceeds the full scale of the port, clamped to 0 dB."
        )
    target_att = max(-power, 0)
    att = min(int(target_att // 2) * 2, max_attenuation)
    delta = target_att - att
    # a change in amplitude by a factor of 10 corresponds to a 20 dB change in level
    offset = float(10.0 ** (-delta / 20.0))
    return att, offset
