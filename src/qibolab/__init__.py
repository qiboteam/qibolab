__version__ = "0.0.1.dev1"

import os
from qibolab.platform import Platform

# TODO: Implement a proper platform selector
platform = Platform(os.environ.get("QIBOLAB_PLATFORM", "tiiq"))
