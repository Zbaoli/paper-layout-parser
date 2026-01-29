"""
PDF Document Layout Detection

This package provides tools for detecting document layout elements
in PDF files using DocLayout-YOLO.
"""

# Core modules
from .core.pdf_converter import PDFConverter
from .core.layout_detector import create_detector, DocLayoutDetector, Detection
from .core.result_processor import ResultProcessor
from .core.figure_extractor import FigureTableExtractor

# Matching modules
from .matching.caption_matcher import CaptionMatcher
from .matching.types import SearchDirection, ExtractedItem, ExtractionResult

# Visualization modules
from .visualization.renderer import BoundingBoxRenderer, create_visualizer
from .visualization.styles import ColorPalette, LabelStrategy, ClassNameLabelStrategy, NumberedLabelStrategy
from .visualization.legend import LegendRenderer

# Backward compatibility alias
Visualizer = BoundingBoxRenderer

__version__ = "2.0.0"
__all__ = [
    # Core
    "PDFConverter",
    "create_detector",
    "DocLayoutDetector",
    "Detection",
    "ResultProcessor",
    "FigureTableExtractor",
    # Matching
    "CaptionMatcher",
    "SearchDirection",
    "ExtractedItem",
    "ExtractionResult",
    # Visualization
    "BoundingBoxRenderer",
    "create_visualizer",
    "ColorPalette",
    "LabelStrategy",
    "ClassNameLabelStrategy",
    "NumberedLabelStrategy",
    "LegendRenderer",
    # Backward compatibility
    "Visualizer",
]
