"""
YOLOv8 PDF Document Layout Detection

This package provides tools for detecting document layout elements
in PDF files using YOLOv8-based models.
"""

from .pdf_converter import PDFConverter
from .layout_detector import create_detector, DocLayoutDetector, YOLOv8LayoutDetector
from .result_processor import ResultProcessor
from .visualizer import Visualizer

__version__ = "1.0.0"
__all__ = [
    "PDFConverter",
    "create_detector",
    "DocLayoutDetector",
    "YOLOv8LayoutDetector",
    "ResultProcessor",
    "Visualizer",
]
