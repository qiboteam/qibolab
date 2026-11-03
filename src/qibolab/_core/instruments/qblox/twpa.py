"""TWPA utility functions for Qblox instruments."""

import warnings

from .limits import MAX_OUTPUT_ATTENUATION


def twpa_attenuation_and_offset(power: float) -> tuple[int, float]:
    """Calculate hardware port attenuation (in dB) and fine-tuning offset factor.

    Follows the standard Qibolab convention, where the power is the opposite of the
    attenuation, relative to the full scale of the port (thus ``power <= 0``).
    The port attenuation is the largest even integer not exceeding the target one, and
    the remainder is applied by the offset, ensuring ``offset <= 1``.

    The port attenuation is clamped to its available range: beyond the maximum, the
    offset applies the whole remainder, while powers above the full scale are clamped to
    it, with a warning.

    Args:
        power: Power in dB, relative to the full scale (conventionally ``<= 0``).

    Returns:
        tuple[int, float]: (attenuation_in_dB, fine_tuning_offset)
    """
    if power > 0:
        warnings.warn(
            f"Power {power} dB exceeds the full scale of the port, clamped to 0 dB."
        )
    target_att = max(-power, 0)
    att = min(int(target_att // 2) * 2, MAX_OUTPUT_ATTENUATION)
    delta = target_att - att
    offset = float(10.0 ** (-delta / 20.0))
    return att, offset
