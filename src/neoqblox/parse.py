from pathlib import Path

from lark import Lark

GRAMMAR_FILE = Path(__file__).parent / "q1asm.lark"
GRAMMAR = GRAMMAR_FILE.read_text()


parser = Lark(GRAMMAR, parser="lalr")


def test(code):
    tree = parser.parse(code)
    print(tree.pretty())


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        test(Path(arg).read_text())
