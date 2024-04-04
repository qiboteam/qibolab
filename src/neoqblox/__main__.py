from pathlib import Path

from .parse import parse

if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        parse(Path(arg).read_text())
