"""
PDF Document Layout Detection

This package provides tools for detecting document layout elements
in PDF files using DocLayout-YOLO.
"""

from .pdf_converter import PDFConverter
from .layout_detector import create_detector, DocLayoutDetector
from .result_processor import ResultProcessor
from .visualizer import Visualizer
from .figure_table_extractor import FigureTableExtractor, ExtractedItem, ExtractionResult

__version__ = "1.0.0"
__all__ = [
    "PDFConverter",
    "create_detector",
    "DocLayoutDetector",
    "ResultProcessor",
    "Visualizer",
    "FigureTableExtractor",
    "ExtractedItem",
    "ExtractionResult",
]
