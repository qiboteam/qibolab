import pytest


def find_instrument(platform, instrument_type):
    for instrument in platform.instruments.values():
        if isinstance(instrument, instrument_type):
            return instrument
    return None


def get_instrument(platform, instrument_type):
    """Finds if an instrument of a given type exists in the given platform.

    If the platform does not have such an instrument, the corresponding
    test that asked for this instrument is skipped. This ensures that
    QPU tests are executed only on the available instruments.
    """
    instrument = find_instrument(platform, instrument_type)
    if instrument is None:
        pytest.skip(f"Skipping {instrument_type.__name__} test for {platform.name}.")
    return instrument
