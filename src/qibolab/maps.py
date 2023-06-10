NameId = str


class NamedType:
    """Arbitrary object that has a name."""

    name: NameId


class NamedMap(dict):
    """Collection of ``NamedType`` objects in a dictionary where the key is their ``name``."""

    Type = NamedType

    def _add_items(self, *items):
        for item in items:
            if isinstance(item, self.Type):
                self[item.name] = item
            else:
                self[item] = self.Type(item)
        return self

    def __ior__(self, items):
        if not isinstance(items, type(self)):
            try:
                if isinstance(items, str):
                    raise TypeError
                items = type(self)()._add_items(*items)
            except TypeError:
                items = type(self)()._add_items(items)
        return super().__ior__(items)
