from qibo.config import log, raise_error
from qibolab.platforms.abstract import AbstractPlatform
# additional contents to be added later on
# abstract paltform will be implemented later on
class cold_ion(AbstractPlatform):
        def _init__(self,name,runcard):
                log.info(f"Loading platform {name} from runcard {runcard}")
                self.name = name
                self.runcard = runcard
                self.is_connected = False
                            
        def run_calibration(self):
                return self.name
#                raise_error(NotImplementedError)
        
