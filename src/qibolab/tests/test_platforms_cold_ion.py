# -*- coding: utf-8 -*-
import numpy as np
import pytest

from qibolab.platform import Platform

# TODO: Modify the platform for the cases required for the cold ion platform.
def test_cold_ion_initialization():
    platform = Platform("cold_ion")
    platform.reload_settings()
    platform.connect()
    platform.setup()
    platform.start()
    platform.stop()
    platform.disconnect()
