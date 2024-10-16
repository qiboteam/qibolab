# Visibility guidelines

Trying to establish some guidelines about importing and "exporting" things

## Internal imports

1. never import from a place which is not supposed to export

   - the export should be contained in `__all__`, but we won't be strict on that, not
     immediately, so the "export places" are either those where the objects are
     initially defined, or on-purpose re-exports from the various `__init__.py`
   - _example:_ if an object is defined in `a.b.c.d.e`, you should import it from there, or from: `a.b.c.d`, `a.b.c`, `a.b`

2. never import from any of your own `__init__.py`

   - `__init__.py` is meant for the export, the first thing encountered by anyone
     looking at the (sub)package from the outside, and it could contain re-exports ->
     high chance of circular imports
   - _example:_ if you are `a.b.c`, never import from `a.b` nor `a` -> no one in Qibolab should import `from qibolab` itself

3. if importing from another subpackage, import at the top-most level possible

   - at least the internals can be changed at will, the compatibility towards the
     outside has only to be honored in a single place
   - _example_: if you are `a.b`, and import from `a.c.d.e`, then import from `a.c`, if it exports it

4. `*` imports are recommended for re-exporting the content of the modules in the
   `__init__.py` above, but they should be avoided for internal usage

   - _re-exports_ should be accompanied by the parent `__all__` update, e.g.
     https://github.com/qiboteam/qibolab/blob/95eab210e44f74bb3866e445efeaffccf5c3bad1/src/qibolab/instruments/qm/components/__init__.py#L1-L5

5. since it seems that the majority of tools understand the `__all__ +=
submodule.__all__` idiom, but not all the list methods (like `.extend()`), and not
   even list comprehension, the redundant syntax above is suggested, namely:

   1. `*` export the content of all the submodules
   2. import all the submodules as module objects, to access their `.__all__` attributes
   3. extend the parent `__all__` with `+=` statements, one at a time

   - it is definitely redundant, since modules are listed three times, but this is to
     ensure the widest compatibility possible with external static tools (i.e. not
     executing the code in a Python interpreter, e.g. linters as pylint & ruff, type
     checkers as mypy & pyright, and others as pycln)

https://github.com/qiboteam/qibolab/pull/1068#discussion_r1796910287

## Module exports

A limit case is whether the only module using something is in the exact same subpackage.
In that case, you would only import from the module itself, and never from the
containing package. In that case, `__all__` is ignored (as in all direct import).

I would suggest to reserve `__all__` for features to be exposed to the user, such that
we can always apply the submodules' `__all__` concatenation strategy in all
`__init__.py`, without any dedicated selection.
And instead using the leading `_` convention to distinguish what is private to a given
module, and what should be available to other internal modules within the package, but
not to the Qibolab's user.

True that, for deeply nested modules, used by other modules very far, it may be
convenient to shorten a bit the import path, by lifting with re-exports the containing
subpackage.
However, for a single level, we can import them manually, doing something like:

```py
# a.py
__all__ = [A, B]

A = 42
B = 43
C = 44
D = 45
_E = 46

# __init__.py
from .a import *
from .a import C
from . import a

__all__ = []
__all__.extend(a.__all__)
```

where `A` and `B` are supposed to be used by the Qibolab user, `D` is separately
shortened, `D` is accessible, and `_E` is private.
