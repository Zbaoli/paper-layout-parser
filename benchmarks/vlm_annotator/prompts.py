"""
VLM Prompt Templates

Contains prompt templates for figure-caption matching analysis.
"""

# System prompt for the VLM
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
