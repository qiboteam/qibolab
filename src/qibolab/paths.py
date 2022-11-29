import os
from pathlib import Path

qibolab_folder = Path(__file__).parent
src_folder = qibolab_folder.parent
project_folder = src_folder.parent
examples_folder = project_folder / "examples"
script_folder = Path(os.path.abspath(""))
user_folder = Path.home() / ".qibolab"
