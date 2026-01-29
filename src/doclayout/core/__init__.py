"""
Core Module

Core PDF processing components for document layout detection.
"""

from .pdf_converter import PDFConverter
from .layout_detector import (
    Detection,
    BaseLayoutDetector,
    DocLayoutDetector,
    create_detector,
    DOCLAYOUT_CLASS_NAMES,
)
from .result_processor import ResultProcessor
from .figure_extractor import FigureTableExtractor

__all__ = [
    # PDF Converter
    "PDFConverter",
    # Layout Detector
    "Detection",
    "BaseLayoutDetector",
    "DocLayoutDetector",
    "create_detector",
    "DOCLAYOUT_CLASS_NAMES",
    # Result Processor
    "ResultProcessor",
    # Figure Extractor
    "FigureTableExtractor",
]
