import yaml
import pytest
import numpy as np 
from qibolab.paths import qibolab_folder
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.platforms.cold_ion_qubit import cold_ion
