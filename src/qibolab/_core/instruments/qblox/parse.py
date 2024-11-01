from pathlib import Path

from lark import Lark, Transformer

from .ast_ import INSTRUCTIONS, Comment, Line, Program

__all__ = []

GRAMMAR_FILE = Path(__file__).parent / "q1asm.lark"
GRAMMAR = GRAMMAR_FILE.read_text()


parser = Lark(GRAMMAR)


class ToAst(Transformer):
    def instruction(self, args):
        name = args[0].data.value
        attrs = (a.value for a in args[0].children)
        return INSTRUCTIONS[name].from_args(*attrs)

    def line(self, args):
        label = args[0].value if args[0] is not None else None
        comment = args[2].value[1:] if args[2] is not None else None
        return Line(instruction=args[1], label=label, comment=comment)

    def comment(self, args):
        return Comment(args[0].value[1:].strip())


def parse(code: str) -> Program:
    """Parse Q1ASM representation."""
    tree = parser.parse(code)
    return Program.from_elements(ToAst().transform(tree).children)
