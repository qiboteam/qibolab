from pathlib import Path

import rich
from lark import Lark

from .ast_ import ToAst

GRAMMAR_FILE = Path(__file__).parent / "q1asm.lark"
GRAMMAR = GRAMMAR_FILE.read_text()


parser = Lark(GRAMMAR)


def parse(code):
    tree = parser.parse(code)
    print(tree.pretty())
    rich.print(ToAst().transform(tree).children)
