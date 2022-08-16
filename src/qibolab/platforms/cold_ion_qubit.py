from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform
# additional contents to be added later on
# abstract paltform will be implemented later on
class cold_ion(AbstractPlatform):
        def run_calibration(self):
                raise_error(NotImplementedError)
