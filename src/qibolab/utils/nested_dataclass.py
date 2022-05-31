"""Nested dataclass decorator."""
from dataclasses import dataclass, is_dataclass
from typing import get_type_hints

import numpy as np

def nested_dataclass(*args, **kwargs):
    """Class decorator used to cast any dict attribute into its corresponding dataclass class."""

    def wrapper(cls):
        """Wrapper that redefines the constructor to cast all attributes to its corresponding classes."""
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = get_type_hints(cls).get(name, None)
                if is_dataclass(field_type):
                    if not isinstance(value, dict):
                        raise ValueError("Using a non-dictionary object as argument to a dataclass.")
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper
