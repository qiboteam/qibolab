from importlib import import_module

import pytest


def load_from_platform(settings, name):
    """Loads instrument from platform, if it is available."""
    for instrument in settings["instruments"].values():
        if instrument["class"] == name:
            lib = instrument["lib"]
            i_class = instrument["class"]
            address = instrument["address"]
            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            return InstrumentClass(name, address), instrument["settings"]
    pytest.skip(f"Skip {name} test as it is not included in the tested platforms.")


class InstrumentsDict(dict):
    def __getitem__(self, name):
        if name not in self:
            pytest.skip(f"Skip {name} test as it is not included in the tested platforms.")
        else:
            return super().__getitem__(name)
