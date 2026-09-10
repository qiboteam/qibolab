"""TWPA utility functions for Qblox instruments."""


def twpa_attenuation_and_offset(power: float) -> tuple[int, float]:
    """Calculate hardware port attenuation (in dB) and fine-tuning offset factor.

    Follows standard Qibolab convention where target attenuation = -power (power <= 0).
    The port attenuation is the largest even integer <= target attenuation,
    ensuring offset <= 1.0.

    Args:
        power: Power in dBm (conventionally <= 0).

    Returns:
        tuple[int, float]: (attenuation_in_dB, fine_tuning_offset)
    """
    target_att = -power if power <= 0 else power
    att = int(target_att // 2) * 2
    delta = target_att - att
    offset = float(10.0 ** (-delta / 20.0))
    return att, offset
