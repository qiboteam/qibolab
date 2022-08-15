# -*- coding: utf-8 -*-
from pathlib import Path
import os

qibolab_folder = Path(__file__).parent
src_folder = qibolab_folder.parent
project_folder = src_folder.parent
examples_folder = project_folder / 'examples'
script_folder = Path(os.path.abspath(''))
