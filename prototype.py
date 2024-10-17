import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Union

d = {
    "a": [1, 2, 3],
    "b": {
        "c": [
            {"d": 42},
        ],
    },
}

u = {
    "b.c[0].d": 37,
    "a[2]": 64,
}

Collection = Union[dict, list]
Path = str
Update = dict[Path, Any]


@dataclass
class Element:
    key: int


@dataclass
class Attribute:
    key: str


Accessor = Union[Element, Attribute]


class TokenKind(Enum):
    BRACKET = r"[\[\]]"
    DOT = r"\."
    ID = r"\w+"

    @classmethod
    def regex(cls) -> str:
        return "|".join(f"(?P<{var.name}>{var.value})" for var in cls)


TOKEN_PATTERN = re.compile(TokenKind.regex())


@dataclass
class Token:
    kind: TokenKind
    value: str


def tokenize(path: Path) -> list[Token]:
    return [
        Token(kind=TokenKind[m.lastgroup], value=m.group())
        for m in TOKEN_PATTERN.finditer(path)
    ]


class State(Enum):
    ATTRIBUTE = auto()
    ELEMENT = auto()
    CLOSE = auto()
    RESET = auto()


def parse(path: list[Token]) -> list[Accessor]:
    accessors = []
    state = State.ATTRIBUTE if path[0].kind is not TokenKind.BRACKET else State.RESET
    for token in path:
        if state in [State.ATTRIBUTE, State.ELEMENT]:
            assert token.kind is TokenKind.ID
            acc = (
                Attribute(key=token.value)
                if state is State.ATTRIBUTE
                else Element(key=int(token.value))
            )
            accessors.append(acc)
            state = State.RESET if state is State.ATTRIBUTE else State.CLOSE
        elif state is State.RESET:
            if token.kind is TokenKind.DOT:
                state = State.ATTRIBUTE
            elif token.kind is TokenKind.BRACKET:
                state = State.ELEMENT
            else:
                assert False
        elif state is State.CLOSE:
            assert token.kind is TokenKind.BRACKET
            state = State.RESET
        else:
            assert False
    if state is not State.RESET:
        assert False
    return accessors


def setvalue(d: Collection, path: Path, val: Any):
    accessors = parse(tokenize(path))
    current = d
    for acc in accessors[:-1]:
        current = current[acc.key]

    current[accessors[-1].key] = val


def update(d: Collection, up: Update):
    for path, val in up.items():
        setvalue(d, path, val)
