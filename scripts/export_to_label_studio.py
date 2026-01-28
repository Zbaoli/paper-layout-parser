#!/usr/bin/env python3
"""
Export VLM annotations to Label Studio format for human correction.

Usage:
    python scripts/export_to_label_studio.py --input data/output --output label_studio_export
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def create_labeling_config() -> str:
    """Create Label Studio labeling config for figure-caption matching."""
    return """
<View>
  <Header value="Figure-Caption Matching Annotation"/>
  <Text name="instructions" value="Review and correct figure-caption matches. Connect figures (F) to their captions (C)."/>

  <Image name="image" value="$image"/>

  <RectangleLabels name="bbox" toName="image">
    <Label value="Figure" background="#FF6B6B"/>
    <Label value="Table" background="#4ECDC4"/>
    <Label value="Caption" background="#FFE66D"/>
  </RectangleLabels>

  <Relations>
    <Relation value="has_caption"/>
  </Relations>
</View>
"""


def convert_bbox_to_percent(bbox: Dict[str, float], img_width: int, img_height: int) -> Dict[str, float]:
    """Convert pixel bbox to percentage for Label Studio."""
    return {
        "x": (bbox["x1"] / img_width) * 100,
        "y": (bbox["y1"] / img_height) * 100,
        "width": ((bbox["x2"] - bbox["x1"]) / img_width) * 100,
        "height": ((bbox["y2"] - bbox["y1"]) / img_height) * 100,
    }


def get_image_dimensions(image_path: Path) -> tuple:
    """Get image dimensions."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except ImportError:
        # Default to common PDF page size at 200 DPI
        return (1654, 2339)


def convert_document(
    doc_dir: Path,
    output_dir: Path,
    images_dir: Path,
) -> List[Dict[str, Any]]:
    """Convert a single document's annotations to Label Studio format."""
    tasks = []

    # Load VLM annotations
    annotation_file = doc_dir / "caption_annotations.json"
    if not annotation_file.exists():
        print(f"  Skipping {doc_dir.name}: no caption_annotations.json")
        return []

    with open(annotation_file) as f:
        annotations = json.load(f)

    pdf_name = annotations.get("pdf_name", doc_dir.name)
    pages_dir = doc_dir / "pages"

    # Process each page with matches
    for page_data in annotations.get("pages", []):
        page_number = page_data.get("page_number", 0)
        matches = page_data.get("matches", [])
        unmatched_figures = page_data.get("unmatched_figures", [])
        unmatched_tables = page_data.get("unmatched_tables", [])
        unmatched_captions = page_data.get("unmatched_captions", [])

        # Skip pages with no annotations
        if not matches and not unmatched_figures and not unmatched_tables:
            continue

        # Find page image
        page_image = None
        for pattern in [f"page_{page_number:04d}.png", f"page_{page_number:03d}.png", f"page_{page_number}.png"]:
            candidate = pages_dir / pattern
            if candidate.exists():
                page_image = candidate
                break

        if not page_image:
            print(f"  Warning: Page image not found for {pdf_name} page {page_number}")
            continue

        # Copy image to output
        rel_image_path = f"{pdf_name}/page_{page_number:04d}.png"
        dest_image = images_dir / rel_image_path
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(page_image, dest_image)

        # Get image dimensions
        img_width, img_height = get_image_dimensions(page_image)

        # Build Label Studio task
        task = {
            "data": {
                "image": f"/data/local-files/?d=images/{rel_image_path}",
                "pdf_name": pdf_name,
                "page_number": page_number,
            },
            "predictions": [{
                "model_version": annotations.get("annotator", "VLM"),
                "result": []
            }]
        }

        result = task["predictions"][0]["result"]
        id_counter = 0
        region_ids = {}  # Map figure/caption IDs to Label Studio region IDs

        # Add matched figures/tables and captions
        for match in matches:
            figure_id = match.get("figure_id", "")
            figure_type = match.get("figure_type", "figure")
            figure_bbox = match.get("figure_bbox")
            caption_id = match.get("caption_id")
            caption_bbox = match.get("caption_bbox")

            # Add figure/table region
            if figure_bbox:
                id_counter += 1
                fig_region_id = f"region_{id_counter}"
                region_ids[figure_id] = fig_region_id

                pct_bbox = convert_bbox_to_percent(figure_bbox, img_width, img_height)
                label = "Table" if figure_type == "table" else "Figure"

                result.append({
                    "id": fig_region_id,
                    "type": "rectanglelabels",
                    "from_name": "bbox",
                    "to_name": "image",
                    "original_width": img_width,
                    "original_height": img_height,
                    "value": {
                        "x": pct_bbox["x"],
                        "y": pct_bbox["y"],
                        "width": pct_bbox["width"],
                        "height": pct_bbox["height"],
                        "rectanglelabels": [label],
                    },
                    "meta": {"text": figure_id}
                })

            # Add caption region
            if caption_bbox and caption_id:
                id_counter += 1
                cap_region_id = f"region_{id_counter}"
                region_ids[caption_id] = cap_region_id

                pct_bbox = convert_bbox_to_percent(caption_bbox, img_width, img_height)

                result.append({
                    "id": cap_region_id,
                    "type": "rectanglelabels",
                    "from_name": "bbox",
                    "to_name": "image",
                    "original_width": img_width,
                    "original_height": img_height,
                    "value": {
                        "x": pct_bbox["x"],
                        "y": pct_bbox["y"],
                        "width": pct_bbox["width"],
                        "height": pct_bbox["height"],
                        "rectanglelabels": ["Caption"],
                    },
                    "meta": {"text": caption_id}
                })

                # Add relation between figure and caption
                if figure_id in region_ids:
                    result.append({
                        "type": "relation",
                        "from_id": region_ids[figure_id],
                        "to_id": cap_region_id,
                        "direction": "right",
                        "labels": ["has_caption"]
                    })

        tasks.append(task)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Export to Label Studio format")
    parser.add_argument("--input", type=str, default="data/output", help="Input directory")
    parser.add_argument("--output", type=str, default="label_studio_export", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from {input_dir} to {output_dir}")

    # Find all documents
    all_tasks = []
    for doc_dir in sorted(input_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        if not (doc_dir / "caption_annotations.json").exists():
            continue

        print(f"Processing: {doc_dir.name}")
        tasks = convert_document(doc_dir, output_dir, images_dir)
        all_tasks.extend(tasks)
        print(f"  Added {len(tasks)} tasks")

    # Save tasks
    tasks_file = output_dir / "tasks.json"
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    # Save labeling config
    config_file = output_dir / "labeling_config.xml"
    with open(config_file, "w") as f:
        f.write(create_labeling_config())

    print(f"\n{'='*50}")
    print(f"Export complete!")
    print(f"{'='*50}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Tasks file: {tasks_file}")
    print(f"Config file: {config_file}")
    print(f"\nNext steps:")
    print(f"1. Install Label Studio: pip install label-studio")
    print(f"2. Start Label Studio: label-studio start")
    print(f"3. Create a new project")
    print(f"4. Go to Settings > Labeling Interface, paste content from {config_file}")
    print(f"5. Import {tasks_file}")
    print(f"6. Set local storage path to: {images_dir.absolute()}")


if __name__ == "__main__":
    main()
