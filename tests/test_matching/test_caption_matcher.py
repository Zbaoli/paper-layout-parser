"""
Tests for CaptionMatcher

Tests the caption matching algorithm against various scenarios.
"""

import pytest
from typing import Dict, List, Any

# Import from new matching module
from src.matching import CaptionMatcher, SearchDirection


class TestCaptionMatcherInit:
    """Tests for CaptionMatcher initialization."""

    def test_default_init(self):
        """Test default initialization."""
        matcher = CaptionMatcher()
        assert matcher.max_vertical_distance == 100.0
        assert matcher.min_horizontal_overlap == 0.3
        assert matcher.figure_search_direction == SearchDirection.BELOW
        assert matcher.table_search_direction == SearchDirection.ABOVE

    def test_custom_init(self):
        """Test custom initialization."""
        matcher = CaptionMatcher(
            max_vertical_distance=150.0,
            min_horizontal_overlap=0.5,
            figure_search_direction=SearchDirection.BOTH,
            table_search_direction=SearchDirection.BELOW,
        )
        assert matcher.max_vertical_distance == 150.0
        assert matcher.min_horizontal_overlap == 0.5
        assert matcher.figure_search_direction == SearchDirection.BOTH
        assert matcher.table_search_direction == SearchDirection.BELOW


class TestHorizontalOverlap:
    """Tests for horizontal overlap calculation."""

    def test_full_overlap(self):
        """Test full horizontal overlap."""
        matcher = CaptionMatcher()
        bbox1 = {"x1": 100.0, "y1": 0.0, "x2": 300.0, "y2": 100.0}
        bbox2 = {"x1": 100.0, "y1": 0.0, "x2": 300.0, "y2": 100.0}
        overlap = matcher._get_horizontal_overlap(bbox1, bbox2)
        assert overlap == 1.0

    def test_partial_overlap(self):
        """Test partial horizontal overlap."""
        matcher = CaptionMatcher()
        bbox1 = {"x1": 100.0, "y1": 0.0, "x2": 300.0, "y2": 100.0}
        bbox2 = {"x1": 200.0, "y1": 0.0, "x2": 400.0, "y2": 100.0}
        overlap = matcher._get_horizontal_overlap(bbox1, bbox2)
        # Overlap is 100px (200-300), smaller width is 200px
        assert overlap == pytest.approx(0.5)

    def test_no_overlap(self):
        """Test no horizontal overlap."""
        matcher = CaptionMatcher()
        bbox1 = {"x1": 100.0, "y1": 0.0, "x2": 200.0, "y2": 100.0}
        bbox2 = {"x1": 300.0, "y1": 0.0, "x2": 400.0, "y2": 100.0}
        overlap = matcher._get_horizontal_overlap(bbox1, bbox2)
        assert overlap == 0.0

    def test_contained_box(self):
        """Test when one box is contained within another horizontally."""
        matcher = CaptionMatcher()
        bbox1 = {"x1": 100.0, "y1": 0.0, "x2": 400.0, "y2": 100.0}
        bbox2 = {"x1": 150.0, "y1": 0.0, "x2": 250.0, "y2": 100.0}
        overlap = matcher._get_horizontal_overlap(bbox1, bbox2)
        # Smaller box is fully contained
        assert overlap == 1.0


class TestVerticalDistance:
    """Tests for vertical distance calculation."""

    def test_caption_below(self):
        """Test caption below item."""
        matcher = CaptionMatcher()
        item_bbox = {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}
        caption_bbox = {"x1": 100.0, "y1": 220.0, "x2": 300.0, "y2": 260.0}

        distance, is_valid = matcher._get_vertical_distance(
            item_bbox, caption_bbox, SearchDirection.BELOW
        )
        assert distance == 20.0
        assert is_valid is True

    def test_caption_above(self):
        """Test caption above item."""
        matcher = CaptionMatcher()
        item_bbox = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
        caption_bbox = {"x1": 100.0, "y1": 140.0, "x2": 300.0, "y2": 180.0}

        distance, is_valid = matcher._get_vertical_distance(
            item_bbox, caption_bbox, SearchDirection.ABOVE
        )
        assert distance == 20.0
        assert is_valid is True

    def test_caption_wrong_direction(self):
        """Test caption in wrong direction."""
        matcher = CaptionMatcher()
        item_bbox = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
        caption_bbox = {"x1": 100.0, "y1": 420.0, "x2": 300.0, "y2": 460.0}

        # Caption is below but we're searching above
        distance, is_valid = matcher._get_vertical_distance(
            item_bbox, caption_bbox, SearchDirection.ABOVE
        )
        assert is_valid is False

    def test_search_both_directions(self):
        """Test searching in both directions."""
        matcher = CaptionMatcher()
        item_bbox = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
        caption_below = {"x1": 100.0, "y1": 420.0, "x2": 300.0, "y2": 460.0}
        caption_above = {"x1": 100.0, "y1": 140.0, "x2": 300.0, "y2": 180.0}

        # Caption below
        distance, is_valid = matcher._get_vertical_distance(
            item_bbox, caption_below, SearchDirection.BOTH
        )
        assert is_valid is True
        assert distance == 20.0

        # Caption above
        distance, is_valid = matcher._get_vertical_distance(
            item_bbox, caption_above, SearchDirection.BOTH
        )
        assert is_valid is True
        assert distance == 20.0


class TestValidMatch:
    """Tests for match validation."""

    def test_valid_match_below(self):
        """Test valid match with caption below."""
        matcher = CaptionMatcher(max_vertical_distance=50.0, min_horizontal_overlap=0.3)
        item_bbox = {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}
        caption_bbox = {"x1": 100.0, "y1": 220.0, "x2": 300.0, "y2": 260.0}

        is_valid, distance = matcher._is_valid_match(
            item_bbox, caption_bbox, SearchDirection.BELOW
        )
        assert is_valid is True
        assert distance == 20.0

    def test_invalid_match_too_far(self):
        """Test invalid match - caption too far."""
        matcher = CaptionMatcher(max_vertical_distance=50.0)
        item_bbox = {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}
        caption_bbox = {"x1": 100.0, "y1": 300.0, "x2": 300.0, "y2": 340.0}

        is_valid, distance = matcher._is_valid_match(
            item_bbox, caption_bbox, SearchDirection.BELOW
        )
        assert is_valid is False

    def test_invalid_match_no_overlap(self):
        """Test invalid match - no horizontal overlap."""
        matcher = CaptionMatcher(min_horizontal_overlap=0.3)
        item_bbox = {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0}
        caption_bbox = {"x1": 300.0, "y1": 220.0, "x2": 400.0, "y2": 260.0}

        is_valid, distance = matcher._is_valid_match(
            item_bbox, caption_bbox, SearchDirection.BELOW
        )
        assert is_valid is False


class TestMatchItemsToCaptions:
    """Tests for the main matching algorithm."""

    def test_single_figure_single_caption(self):
        """Test matching single figure with single caption."""
        matcher = CaptionMatcher()
        figures = [{"bbox": {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}}]
        captions = [{"bbox": {"x1": 100.0, "y1": 220.0, "x2": 300.0, "y2": 260.0}}]

        matches = matcher.match_items_to_captions(figures, captions, "figure")

        assert len(matches) == 1
        assert matches[0][0] == figures[0]
        assert matches[0][1] == captions[0]

    def test_figure_no_caption(self):
        """Test figure without matching caption."""
        matcher = CaptionMatcher()
        figures = [{"bbox": {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}}]
        captions = []

        matches = matcher.match_items_to_captions(figures, captions, "figure")

        assert len(matches) == 1
        assert matches[0][0] == figures[0]
        assert matches[0][1] is None

    def test_multiple_figures_captions(self):
        """Test multiple figures and captions on same page."""
        matcher = CaptionMatcher()
        figures = [
            {"bbox": {"x1": 50.0, "y1": 100.0, "x2": 250.0, "y2": 200.0}},
            {"bbox": {"x1": 300.0, "y1": 100.0, "x2": 500.0, "y2": 200.0}},
        ]
        captions = [
            {"bbox": {"x1": 50.0, "y1": 220.0, "x2": 250.0, "y2": 260.0}},
            {"bbox": {"x1": 300.0, "y1": 220.0, "x2": 500.0, "y2": 260.0}},
        ]

        matches = matcher.match_items_to_captions(figures, captions, "figure")

        assert len(matches) == 2
        # Each figure should match its corresponding caption
        assert matches[0][0] == figures[0]
        assert matches[0][1] == captions[0]
        assert matches[1][0] == figures[1]
        assert matches[1][1] == captions[1]

    def test_table_caption_above(self):
        """Test table with caption above (default for tables)."""
        matcher = CaptionMatcher()
        tables = [{"bbox": {"x1": 100.0, "y1": 200.0, "x2": 400.0, "y2": 500.0}}]
        captions = [{"bbox": {"x1": 100.0, "y1": 150.0, "x2": 400.0, "y2": 180.0}}]

        matches = matcher.match_items_to_captions(tables, captions, "table")

        assert len(matches) == 1
        assert matches[0][0] == tables[0]
        assert matches[0][1] == captions[0]

    def test_greedy_matching_closer_wins(self):
        """Test that greedy matching prefers closer captions."""
        matcher = CaptionMatcher()
        figures = [{"bbox": {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}}]
        captions = [
            {"bbox": {"x1": 100.0, "y1": 250.0, "x2": 300.0, "y2": 290.0}},  # Farther
            {"bbox": {"x1": 100.0, "y1": 210.0, "x2": 300.0, "y2": 240.0}},  # Closer
        ]

        matches = matcher.match_items_to_captions(figures, captions, "figure")

        assert len(matches) == 1
        # Should match the closer caption
        assert matches[0][1] == captions[1]

    def test_empty_items(self):
        """Test with empty items list."""
        matcher = CaptionMatcher()
        matches = matcher.match_items_to_captions([], [], "figure")
        assert matches == []

    def test_caption_cannot_match_multiple_figures(self):
        """Test that each caption can only match one figure."""
        matcher = CaptionMatcher()
        # Two figures close together
        figures = [
            {"bbox": {"x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 200.0}},
            {"bbox": {"x1": 100.0, "y1": 220.0, "x2": 250.0, "y2": 320.0}},
        ]
        # One caption between them - will match closer figure
        captions = [
            {"bbox": {"x1": 100.0, "y1": 340.0, "x2": 250.0, "y2": 370.0}},
        ]

        matches = matcher.match_items_to_captions(figures, captions, "figure")

        assert len(matches) == 2
        # Only one figure should have a caption
        matched_captions = [m[1] for m in matches if m[1] is not None]
        assert len(matched_captions) == 1


class TestSearchDirection:
    """Tests for SearchDirection enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert SearchDirection.BELOW.value == "below"
        assert SearchDirection.ABOVE.value == "above"
        assert SearchDirection.BOTH.value == "both"


class TestClassConstants:
    """Tests for class constants."""

    def test_figure_classes(self):
        """Test FIGURE_CLASSES constant."""
        assert "Figure" in CaptionMatcher.FIGURE_CLASSES

    def test_table_classes(self):
        """Test TABLE_CLASSES constant."""
        assert "Table" in CaptionMatcher.TABLE_CLASSES

    def test_figure_caption_classes(self):
        """Test FIGURE_CAPTION_CLASSES constant."""
        assert "Figure-Caption" in CaptionMatcher.FIGURE_CAPTION_CLASSES

    def test_table_caption_classes(self):
        """Test TABLE_CAPTION_CLASSES constant."""
        assert "Table-Caption" in CaptionMatcher.TABLE_CAPTION_CLASSES
