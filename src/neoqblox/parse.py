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

    test(Path(sys.argv[1]).read_text())
