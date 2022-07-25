# from __future__ import absolute_import
# import numpy as np
# from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
# from qibolab.instruments.icarusqfpga import AbstractInstrument
# from qibolab.instruments.ad9959 import AD9959

import yaml
import pytest
import numpy as np 
from qibolab.paths import qibolab_folder
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.instruments.dds_ad9959 import AD9959


evaluation_board = AD9959()
evaluation_board.set_clock_multiplier(15)
evaluation_board.set_frequency(200e6,channel=[0])
