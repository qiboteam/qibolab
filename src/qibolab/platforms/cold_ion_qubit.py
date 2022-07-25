from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform


class cold_ion(AbstractPlatform):
        def __init__(self, name, runcard):
         super().__init__(name, runcard)
 
         pid = self.settings.get("pid")
         vid = self.settings.get("vid")