"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import pytest
from typing import Dict, List, Any


@pytest.fixture
def sample_bbox() -> Dict[str, float]:
    """Sample bounding box for testing."""
    return {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}


@pytest.fixture
def sample_figure_detection(sample_bbox) -> Dict[str, Any]:
    """Sample figure detection."""
    return {
        "class_id": 3,
        "class_name": "Figure",
        "confidence": 0.95,
        "bbox": sample_bbox,
    }


@pytest.fixture
def sample_caption_detection() -> Dict[str, Any]:
    """Sample caption detection below a figure."""
    return {
        "class_id": 4,
        "class_name": "Figure-Caption",
        "confidence": 0.90,
        "bbox": {"x1": 100.0, "y1": 420.0, "x2": 300.0, "y2": 460.0},
    }


@pytest.fixture
def sample_table_detection() -> Dict[str, Any]:
    """Sample table detection."""
    return {
        "class_id": 5,
        "class_name": "Table",
        "confidence": 0.92,
        "bbox": {"x1": 50.0, "y1": 500.0, "x2": 400.0, "y2": 700.0},
    }


@pytest.fixture
def sample_table_caption_detection() -> Dict[str, Any]:
    """Sample table caption detection above a table."""
    return {
        "class_id": 6,
        "class_name": "Table-Caption",
        "confidence": 0.88,
        "bbox": {"x1": 50.0, "y1": 460.0, "x2": 400.0, "y2": 490.0},
    }


@pytest.fixture
def sample_page_detections(
    sample_figure_detection,
    sample_caption_detection,
    sample_table_detection,
    sample_table_caption_detection,
) -> List[Dict[str, Any]]:
    """Sample page with multiple detections."""
    return [
        sample_figure_detection,
        sample_caption_detection,
        sample_table_detection,
        sample_table_caption_detection,
    ]
