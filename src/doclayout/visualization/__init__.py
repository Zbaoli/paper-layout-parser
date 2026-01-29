"""
Visualization Module

Provides unified rendering capabilities for bounding box visualization
with support for different labeling strategies.
"""

from .renderer import BoundingBoxRenderer, create_visualizer
from .styles import (
    ColorPalette,
    LabelStrategy,
    ClassNameLabelStrategy,
    NumberedLabelStrategy,
    DEFAULT_COLORS,
)
from .legend import LegendRenderer

__all__ = [
    # Renderer
    "BoundingBoxRenderer",
    "create_visualizer",
    # Styles
    "ColorPalette",
    "LabelStrategy",
    "ClassNameLabelStrategy",
    "NumberedLabelStrategy",
    "DEFAULT_COLORS",
    # Legend
    "LegendRenderer",
]
