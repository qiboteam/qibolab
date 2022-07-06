"""BusElement class"""
from qibolab.typings.enums import (
    InstrumentName,
    NodeName,
    PulseShapeName,
    ResultName,
    SystemControlSubcategory,
)


class FactoryElement:
    """Class FactoryElement. All factory element classes must inherit from this class."""

    name: SystemControlSubcategory | PulseShapeName | ResultName | InstrumentName | NodeName
