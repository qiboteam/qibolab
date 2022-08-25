# -*- coding: utf-8 -*-
import numpy as np
import pytest

from qibolab.platform import Platform

# TODO: Modify the platform for the cases required for the cold ion platform initialisation
def test_cold_ion_initialization(): #pragma: no cover
    platform = Platform("cold_ion")
    platform.reload_settings()
    platform.run_calibration()
    platform.connect()
    platform.start()
    platform.stop()
    platform.disconnect()
