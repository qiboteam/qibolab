import yaml
import pytest
import numpy as np 
from qibolab.paths import qibolab_folder
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.platforms.cold_ion_qubit import cold_ion
from qibolab.instruments.dds_ad9959 import AD9959

evaluation_board = AD9959()
evaluation_board.set_clock_multiplier(20)
evaluation_board.set_frequency(20e6, channel=[0])
evaluation_board.set_frequency(200e6,channel=[0])