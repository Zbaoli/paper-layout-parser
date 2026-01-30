"""
VLM Prompt Templates

Contains prompt templates for figure-caption matching analysis.
"""

# =============================================================================
# Detection-based annotation prompts (legacy mode)
# =============================================================================

# System prompt for the VLM (detection-based mode)
SYSTEM_PROMPT = """You are an expert document analysis assistant specialized in \
understanding academic papers and technical documents. Your task is to analyze \
document pages and identify which captions belong to which figures or tables.

You will receive:
1. An image of a document page with annotated bounding boxes
2. Information about the detected elements

The bounding boxes are color-coded and labeled:
- GREEN boxes labeled F1, F2, F3... are FIGURES (images, charts, diagrams)
- BLUE boxes labeled T1, T2, T3... are TABLES
- ORANGE boxes labeled C1, C2, C3... are CAPTIONS (figure captions, table captions)

Your task is to determine which caption (C#) corresponds to which figure (F#) or table (T#).

Guidelines for matching:
1. Captions are typically positioned directly below or above their corresponding figure/table
2. Caption text usually starts with "Figure X", "Fig. X", "Table X", or "Tab. X"
3. The caption content should semantically relate to the figure/table it describes
4. A figure/table may have no caption (especially if it spans multiple pages)
5. A caption may have no corresponding figure/table (if the figure is on another page)

Be conservative: if you're not confident about a match, indicate lower confidence."""

# =============================================================================
# Direct annotation prompts (no pre-detection required)
# =============================================================================

DIRECT_SYSTEM_PROMPT = """You are a document analysis expert. Your task is to analyze \
document pages to identify all Figures, Tables, and their Captions, then establish \
matching relationships between them.

You work directly with raw document page images without any pre-processing or detection.
Your analysis serves as ground truth for evaluating automated detection systems."""

# User prompt template
USER_PROMPT_TEMPLATE = """Analyze this document page and match figures/tables to their captions.

Detected elements on this page:
{elements_description}

Please respond with a JSON object containing:
1. "matches": array of objects with:
   - "figure_id": the figure number (1 for F1, 2 for F2, etc.) or null
   - "figure_type": "figure" or "table"
   - "caption_id": the caption number (1 for C1, 2 for C2, etc.) or null if no match
   - "confidence": your confidence level (0.0 to 1.0)
   - "reasoning": brief explanation of why you made this match

2. "unmatched_captions": array of caption IDs that don't match any figure/table on this page

Example response:
{{
  "matches": [
    {{"figure_id": 1, "figure_type": "figure", "caption_id": 1, "confidence": 0.95,
      "reasoning": "C1 is directly below F1 and starts with 'Figure 1'"}},
    {{"figure_id": 2, "figure_type": "figure", "caption_id": null, "confidence": 0.8,
      "reasoning": "F2 appears to be a continuation, caption likely on previous page"}},
    {{"figure_id": 1, "figure_type": "table", "caption_id": 2, "confidence": 0.9,
      "reasoning": "C2 is above T1 and says 'Table 1'"}}
  ],
  "unmatched_captions": [3]
}}

Respond ONLY with the JSON object, no additional text."""


def format_elements_description(
    figures: list,
    tables: list,
    captions: list,
) -> str:
    """
    Format the elements description for the prompt.

    Args:
        figures: List of figure detections with id
        tables: List of table detections with id
        captions: List of caption detections with id and optional text

    Returns:
        Formatted string describing the elements
    """
    lines = []

    if figures:
        lines.append("FIGURES (green boxes):")
        for fig in figures:
            x1, y1 = fig["bbox"]["x1"], fig["bbox"]["y1"]
            lines.append(f"  - F{fig['id']}: at position ({x1:.0f}, {y1:.0f})")

    if tables:
        lines.append("TABLES (blue boxes):")
        for tbl in tables:
            x1, y1 = tbl["bbox"]["x1"], tbl["bbox"]["y1"]
            lines.append(f"  - T{tbl['id']}: at position ({x1:.0f}, {y1:.0f})")

    if captions:
        lines.append("CAPTIONS (orange boxes):")
        for cap in captions:
            text = cap.get("text", "")
            if len(text) > 50:
                text_preview = text[:50] + "..."
            else:
                text_preview = text or "[no text]"
            x1, y1 = cap["bbox"]["x1"], cap["bbox"]["y1"]
            lines.append(f"  - C{cap['id']}: \"{text_preview}\" at ({x1:.0f}, {y1:.0f})")

    if not lines:
        lines.append("No figures, tables, or captions detected on this page.")

    return "\n".join(lines)


def build_user_prompt(
    figures: list,
    tables: list,
    captions: list,
) -> str:
    """
    Build the complete user prompt for VLM analysis.

    Args:
        figures: List of figure detections
        tables: List of table detections
        captions: List of caption detections

    Returns:
        Complete user prompt string
    """
    elements_desc = format_elements_description(figures, tables, captions)
    return USER_PROMPT_TEMPLATE.format(elements_description=elements_desc)


# =============================================================================
# Direct annotation prompt builder
# =============================================================================

DIRECT_USER_PROMPT = """Analyze this document page and complete the following tasks:

1. Identify all visual elements:
   - Figure: images, charts, diagrams, plots, photographs, illustrations, etc.
   - Table: data tables with rows and columns
   - Caption: figure or table captions (typically starting with "Figure X", "Fig. X", \
"Table X", "Tab. X", or similar patterns)

2. For each element, provide its bounding box:
   - Use normalized coordinates in range 0-1000
   - Format: {"x1": left, "y1": top, "x2": right, "y2": bottom}
   - Coordinates are relative to image dimensions (0=top/left edge, 1000=bottom/right edge)

3. Establish matching relationships:
   - Pair each Figure/Table with its corresponding Caption
   - If a Figure/Table has no Caption on this page, mark it as unmatched
   - If a Caption has no corresponding Figure/Table on this page, mark it as unmatched

Important guidelines:
- Captions are typically positioned directly below or above their corresponding figure/table
- Caption text usually contains a numbering pattern like "Figure 1", "Fig. 1", "Table 1"
- Be thorough: identify ALL figures, tables, and captions on the page
- Be accurate: only create matches when you are confident about the relationship
- Provide accurate bounding boxes that tightly enclose each element

Output in JSON format:
{
  "elements": [
    {"id": 1, "type": "figure", "description": "Bar chart showing experimental results", "bbox": {"x1": 100, "y1": 150, "x2": 900, "y2": 500}},
    {"id": 2, "type": "caption", "text": "Figure 1: Experimental results comparison", "bbox": {"x1": 100, "y1": 510, "x2": 900, "y2": 550}},
    {"id": 3, "type": "table", "description": "Data summary table with 5 columns", "bbox": {"x1": 50, "y1": 600, "x2": 950, "y2": 850}},
    {"id": 4, "type": "caption", "text": "Table 1: Summary of experimental data", "bbox": {"x1": 50, "y1": 860, "x2": 950, "y2": 900}}
  ],
  "matches": [
    {"figure_id": 1, "figure_type": "figure", "caption_id": 2},
    {"figure_id": 3, "figure_type": "table", "caption_id": 4}
  ],
  "unmatched_figures": [],
  "unmatched_tables": [],
  "unmatched_captions": []
}

Notes:
- Element IDs should be unique integers starting from 1
- Each element MUST include a "bbox" field with normalized coordinates (0-1000)
- In "matches", figure_id refers to the element ID of a figure or table
- In "matches", figure_type should be "figure" or "table"
- In "matches", caption_id refers to the element ID of the matched caption
- Unmatched lists contain element IDs of items without matches on this page

Respond ONLY with the JSON object, no additional text."""


def build_direct_user_prompt() -> str:
    """
    Build the user prompt for direct VLM analysis (no pre-detection).

    Returns:
        Complete user prompt string for direct annotation mode
    """
    return DIRECT_USER_PROMPT
