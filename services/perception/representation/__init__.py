"""Minimum-sufficient representation selection for perception outputs."""

from services.perception.representation.mobilenet import MobileNetV3SmallSelector
from services.perception.representation.selector import (
    HeuristicRepresentationSelector,
    RepresentationSelector,
)

__all__ = [
    "HeuristicRepresentationSelector",
    "MobileNetV3SmallSelector",
    "RepresentationSelector",
]
