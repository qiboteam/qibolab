import numpy as np
import qibolab.instruments
evaluation_board = qibolab.instruments.AD9959()
evaluation_board.set_clock_multiplier(15)
evaluation_board.set_frequency(200e6,channel=[0])
