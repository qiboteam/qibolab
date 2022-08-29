# -*- coding: utf-8 -*-
import numpy as np
import pytest
import yaml

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.instruments.dds_ad9959 import AD9959
from qibolab.paths import qibolab_folder
from qibolab.platforms.cold_ion_qubit import cold_ion

# evaluation_board = AD9959()
# evaluation_board.set_clock_multiplier(20)
# evaluation_board.set_frequency(20e6, channel=[0])
# evaluation_board.set_frequency(200e6, channel=[0])
