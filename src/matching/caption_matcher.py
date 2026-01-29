"""
Caption Matcher

Matches figures/tables with their captions based on spatial proximity.
"""

from typing import Any, Dict, List, Optional, Tuple

from .types import SearchDirection


class CaptionMatcher:
    """Matches figures/tables with their captions based on spatial proximity."""

    # Class name mappings for DocLayout-YOLO
    FIGURE_CLASSES = {"Figure"}
    TABLE_CLASSES = {"Table"}
    FIGURE_CAPTION_CLASSES = {"Figure-Caption", "Figure-caption"}
    TABLE_CAPTION_CLASSES = {"Table-Caption", "Table-caption"}

    def __init__(
        self,
        max_vertical_distance: float = 100.0,
        min_horizontal_overlap: float = 0.3,
        figure_search_direction: SearchDirection = SearchDirection.BELOW,
        table_search_direction: SearchDirection = SearchDirection.ABOVE,
    ):
        """
        Initialize the caption matcher.

        Args:
            max_vertical_distance: Maximum vertical distance in pixels between
                                   item and caption (default 100px at 200 DPI ~ 0.5 inch)
            min_horizontal_overlap: Minimum horizontal overlap ratio (0-1)
            figure_search_direction: Direction to search for figure captions
            table_search_direction: Direction to search for table captions
        """
        self.max_vertical_distance = max_vertical_distance
        self.min_horizontal_overlap = min_horizontal_overlap
        self.figure_search_direction = figure_search_direction
        self.table_search_direction = table_search_direction

    def _get_horizontal_overlap(
        self,
        bbox1: Dict[str, float],
        bbox2: Dict[str, float],
    ) -> float:
        """Calculate the horizontal overlap ratio between two bounding boxes."""
        x1_min, x1_max = bbox1["x1"], bbox1["x2"]
        x2_min, x2_max = bbox2["x1"], bbox2["x2"]

        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        overlap = max(0, overlap_end - overlap_start)

        # Calculate overlap relative to the smaller width
        width1 = x1_max - x1_min
        width2 = x2_max - x2_min
        min_width = min(width1, width2)

        if min_width <= 0:
            return 0.0

        return overlap / min_width

    def _get_vertical_distance(
        self,
        item_bbox: Dict[str, float],
        caption_bbox: Dict[str, float],
        search_direction: SearchDirection,
    ) -> Tuple[float, bool]:
        """
        Calculate vertical distance between item and caption.

        Args:
            item_bbox: Bounding box of the item (figure/table)
            caption_bbox: Bounding box of the caption
            search_direction: Direction to search for caption

        Returns:
            Tuple of (distance, is_valid_direction) where:
            - distance: Absolute vertical distance in pixels
            - is_valid_direction: Whether caption is in the valid direction
        """
        # Distance when caption is below item (caption top - item bottom)
        dist_below = caption_bbox["y1"] - item_bbox["y2"]
        # Distance when caption is above item (item top - caption bottom)
        dist_above = item_bbox["y1"] - caption_bbox["y2"]

        if search_direction == SearchDirection.BELOW:
            return abs(dist_below), dist_below >= 0
        elif search_direction == SearchDirection.ABOVE:
            return abs(dist_above), dist_above >= 0
        else:  # BOTH
            if dist_below >= 0:
                return dist_below, True
            elif dist_above >= 0:
                return dist_above, True
            # Overlapping case - use minimum distance
            return 0.0, True

    def _is_valid_match(
        self,
        item_bbox: Dict[str, float],
        caption_bbox: Dict[str, float],
        search_direction: SearchDirection,
    ) -> Tuple[bool, float]:
        """
        Check if a caption is a valid match for an item.

        Args:
            item_bbox: Bounding box of the item (figure/table)
            caption_bbox: Bounding box of the caption
            search_direction: Direction to search for caption

        Returns:
            Tuple of (is_valid, distance) where distance is used for ranking
        """
        # Check vertical distance and direction
        vertical_distance, is_valid_direction = self._get_vertical_distance(
            item_bbox, caption_bbox, search_direction
        )
        if not is_valid_direction or vertical_distance > self.max_vertical_distance:
            return False, float("inf")

        # Check horizontal overlap
        overlap = self._get_horizontal_overlap(item_bbox, caption_bbox)
        if overlap < self.min_horizontal_overlap:
            return False, float("inf")

        return True, vertical_distance

    def match_items_to_captions(
        self,
        items: List[Dict[str, Any]],
        captions: List[Dict[str, Any]],
        item_type: str = "figure",
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Match items (figures/tables) to their captions.

        Uses greedy matching: each caption can only be matched to one item,
        and we prefer closer matches.

        Args:
            items: List of detection dicts with "bbox" key
            captions: List of detection dicts with "bbox" key
            item_type: Type of items being matched ("figure" or "table")

        Returns:
            List of (item, caption) tuples, where caption may be None
        """
        if not items:
            return []

        if not captions:
            return [(item, None) for item in items]

        # Select search direction based on item type
        if item_type == "table":
            search_direction = self.table_search_direction
        else:
            search_direction = self.figure_search_direction

        # Calculate all valid matches with distances
        matches = []
        for item in items:
            for caption in captions:
                is_valid, distance = self._is_valid_match(
                    item["bbox"], caption["bbox"], search_direction
                )
                if is_valid:
                    matches.append((item, caption, distance))

        # Sort by distance (prefer closer matches)
        matches.sort(key=lambda x: x[2])

        # Greedy matching
        used_items = set()
        used_captions = set()
        result_matches = {}

        for item, caption, _ in matches:
            item_id = id(item)
            caption_id = id(caption)

            if item_id not in used_items and caption_id not in used_captions:
                result_matches[item_id] = caption
                used_items.add(item_id)
                used_captions.add(caption_id)

        # Build result list
        result = []
        for item in items:
            item_id = id(item)
            caption = result_matches.get(item_id)
            result.append((item, caption))

        return result
