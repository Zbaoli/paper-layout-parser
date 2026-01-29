#!/usr/bin/env python3
"""
Export VLM annotations to Label Studio format for human correction.

Supports two storage modes:
1. Local storage (default): Images stored locally for Label Studio local file serving
2. MinIO storage: Images uploaded to MinIO for cloud-based access

Usage:
    # Local storage
    python scripts/export_to_label_studio.py --input data/output --output label_studio_export

    # MinIO storage
    python scripts/export_to_label_studio.py --input data/output --output label_studio_export \
        --storage minio --bucket label-studio
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class MinIOStorage:
    """MinIO storage backend for Label Studio images."""

    def __init__(self, bucket: str):
        from minio import Minio

        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self.endpoint = endpoint
        self.secure = secure

        # Ensure bucket exists
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
            print(f"Created MinIO bucket: {bucket}")

    def upload(self, local_path: Path, remote_path: str) -> str:
        """Upload file to MinIO and return public URL."""
        self.client.fput_object(
            self.bucket,
            remote_path,
            str(local_path),
            content_type="image/png",
        )
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}/{self.bucket}/{remote_path}"

    def get_label_studio_url(self, remote_path: str) -> str:
        """Get URL for Label Studio (using MinIO as S3 storage)."""
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}/{self.bucket}/{remote_path}"


class LocalStorage:
    """Local storage backend for Label Studio images."""

    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: Path, remote_path: str) -> str:
        """Copy file to local images directory."""
        dest = self.images_dir / remote_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dest)
        return f"/data/local-files/?d=images/{remote_path}"

    def get_label_studio_url(self, remote_path: str) -> str:
        """Get URL for Label Studio local file serving."""
        return f"/data/local-files/?d=images/{remote_path}"


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


def convert_bbox_to_percent(
    bbox: Dict[str, float], img_width: int, img_height: int
) -> Dict[str, float]:
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


def build_bbox_lookup(doc_dir: Path) -> Dict[str, Dict]:
    """
    Build a lookup table for all figure/table/caption bboxes from extraction_metadata.json.

    Returns dict mapping item_id -> {bbox, caption_bbox, item_type, page_number}
    """
    lookup = {}

    # Try extraction_metadata.json first (has matched pairs)
    extraction_file = doc_dir / "extractions" / "extraction_metadata.json"
    if extraction_file.exists():
        with open(extraction_file) as f:
            data = json.load(f)

        for item in data.get("figures", []):
            item_id = item.get("item_id")
            if item_id:
                lookup[item_id] = {
                    "bbox": item.get("item_bbox"),
                    "caption_bbox": item.get("caption_bbox"),
                    "item_type": "figure",
                    "page_number": item.get("page_number"),
                }

        for item in data.get("tables", []):
            item_id = item.get("item_id")
            if item_id:
                lookup[item_id] = {
                    "bbox": item.get("item_bbox"),
                    "caption_bbox": item.get("caption_bbox"),
                    "item_type": "table",
                    "page_number": item.get("page_number"),
                }

    # Also load result.json to get all detections (including unmatched captions)
    result_file = doc_dir / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            result_data = json.load(f)

        for page in result_data.get("pages", []):
            page_num = page.get("page_number", 0)
            fig_count = 0
            table_count = 0
            cap_count = 0

            for det in page.get("detections", []):
                class_name = det.get("class_name", "")
                bbox = det.get("bbox")

                if class_name == "Figure":
                    fig_count += 1
                    item_id = f"fig_{page_num:02d}_{fig_count:02d}"
                    if item_id not in lookup:
                        lookup[item_id] = {
                            "bbox": bbox,
                            "item_type": "figure",
                            "page_number": page_num,
                        }
                elif class_name == "Table":
                    table_count += 1
                    item_id = f"table_{page_num:02d}_{table_count:02d}"
                    if item_id not in lookup:
                        lookup[item_id] = {
                            "bbox": bbox,
                            "item_type": "table",
                            "page_number": page_num,
                        }
                elif class_name in ("Figure-Caption", "Table-Caption"):
                    cap_count += 1
                    item_id = f"cap_{page_num:02d}_{cap_count:02d}"
                    if item_id not in lookup:
                        lookup[item_id] = {
                            "bbox": bbox,
                            "item_type": "caption",
                            "page_number": page_num,
                        }

    return lookup


def convert_document(
    doc_dir: Path,
    storage: Any,
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

    # Build bbox lookup from extraction_metadata.json and result.json
    bbox_lookup = build_bbox_lookup(doc_dir)

    pdf_name = annotations.get("pdf_name", doc_dir.name)
    pages_dir = doc_dir / "pages"

    # Process each page
    for page_data in annotations.get("pages", []):
        page_number = page_data.get("page_number", 0)
        matches = page_data.get("matches", [])
        unmatched_figures = page_data.get("unmatched_figures", [])
        unmatched_tables = page_data.get("unmatched_tables", [])
        unmatched_captions = page_data.get("unmatched_captions", [])

        # Skip pages with no annotations
        if (
            not matches
            and not unmatched_figures
            and not unmatched_tables
            and not unmatched_captions
        ):
            continue

        # Find page image
        page_image = None
        for pattern in [
            f"page_{page_number:04d}.png",
            f"page_{page_number:03d}.png",
            f"page_{page_number}.png",
        ]:
            candidate = pages_dir / pattern
            if candidate.exists():
                page_image = candidate
                break

        if not page_image:
            print(f"  Warning: Page image not found for {pdf_name} page {page_number}")
            continue

        # Upload/copy image to storage
        rel_image_path = f"{pdf_name}/page_{page_number:04d}.png"
        image_url = storage.upload(page_image, rel_image_path)

        # Get image dimensions
        img_width, img_height = get_image_dimensions(page_image)

        # Build Label Studio task
        task = {
            "data": {
                "image": image_url,
                "pdf_name": pdf_name,
                "page_number": page_number,
            },
            "predictions": [
                {"model_version": annotations.get("annotator", "VLM"), "result": []}
            ],
        }

        result = task["predictions"][0]["result"]
        id_counter = 0
        region_ids = {}  # Map figure/caption IDs to Label Studio region IDs

        # Helper function to add a region
        def add_region(item_id: str, bbox: Dict, label: str) -> Optional[str]:
            nonlocal id_counter
            if not bbox:
                return None

            id_counter += 1
            region_id = f"region_{id_counter}"
            region_ids[item_id] = region_id

            pct_bbox = convert_bbox_to_percent(bbox, img_width, img_height)

            result.append(
                {
                    "id": region_id,
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
                    "meta": {"text": item_id},
                }
            )
            return region_id

        # Helper function to add a relation
        def add_relation(from_id: str, to_id: str):
            if from_id and to_id:
                result.append(
                    {
                        "type": "relation",
                        "from_id": from_id,
                        "to_id": to_id,
                        "direction": "right",
                        "labels": ["has_caption"],
                    }
                )

        # Process matched figures/tables and captions
        for match in matches:
            figure_id = match.get("figure_id", "")
            figure_type = match.get("figure_type", "figure")
            figure_bbox = match.get("figure_bbox")
            caption_id = match.get("caption_id")
            caption_bbox = match.get("caption_bbox")

            # Add figure/table region
            label = "Table" if figure_type == "table" else "Figure"
            fig_region_id = add_region(figure_id, figure_bbox, label)

            # Add caption region
            cap_region_id = None
            if caption_id and caption_bbox:
                cap_region_id = add_region(caption_id, caption_bbox, "Caption")

            # Add relation
            if fig_region_id and cap_region_id:
                add_relation(fig_region_id, cap_region_id)

        # Process unmatched figures (using bbox from lookup)
        for fig_id in unmatched_figures:
            if fig_id in region_ids:
                continue  # Already added
            info = bbox_lookup.get(fig_id, {})
            bbox = info.get("bbox")
            if bbox:
                add_region(fig_id, bbox, "Figure")

        # Process unmatched tables (using bbox from lookup)
        for table_id in unmatched_tables:
            if table_id in region_ids:
                continue  # Already added
            info = bbox_lookup.get(table_id, {})
            bbox = info.get("bbox")
            if bbox:
                add_region(table_id, bbox, "Table")

        # Process unmatched captions (using bbox from lookup)
        for cap_id in unmatched_captions:
            if cap_id in region_ids:
                continue  # Already added
            info = bbox_lookup.get(cap_id, {})
            bbox = info.get("bbox")
            if bbox:
                add_region(cap_id, bbox, "Caption")

        if result:  # Only add task if it has regions
            tasks.append(task)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Export to Label Studio format")
    parser.add_argument(
        "--input", type=str, default="data/output", help="Input directory"
    )
    parser.add_argument(
        "--output", type=str, default="label_studio_export", help="Output directory"
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=["local", "minio"],
        default="local",
        help="Storage backend: local or minio (default: local)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="label-studio",
        help="MinIO bucket name (only used with --storage minio)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from {input_dir} to {output_dir}")
    print(f"Storage backend: {args.storage}")

    # Initialize storage backend
    if args.storage == "minio":
        storage = MinIOStorage(bucket=args.bucket)
        print(f"MinIO bucket: {args.bucket}")
    else:
        images_dir = output_dir / "images"
        storage = LocalStorage(images_dir=images_dir)

    # Find all documents
    all_tasks = []
    for doc_dir in sorted(input_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        if not (doc_dir / "caption_annotations.json").exists():
            continue

        print(f"Processing: {doc_dir.name}")
        tasks = convert_document(doc_dir, storage)
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
    print("Export complete!")
    print(f"{'='*50}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Tasks file: {tasks_file}")
    print(f"Config file: {config_file}")

    if args.storage == "minio":
        print(f"\nMinIO storage instructions:")
        print(f"1. Ensure MinIO is running and accessible")
        print(f"2. Images are stored in bucket: {args.bucket}")
        print(f"3. In Label Studio, configure Cloud Storage:")
        print(f"   - Go to Project Settings > Cloud Storage > Add Source Storage")
        print(f"   - Select 'Amazon S3' (MinIO is S3-compatible)")
        print(f"   - Bucket: {args.bucket}")
        print(f"   - Set endpoint URL to your MinIO server")
        print(f"4. Import {tasks_file}")
    else:
        print(f"\nLocal storage instructions:")
        print(f"1. Start Label Studio with local file serving:")
        print(f"   export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"   export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={output_dir.absolute()}")
        print(f"   label-studio start")
        print(f"2. Create a new project")
        print(f"3. Go to Settings > Labeling Interface, paste content from {config_file}")
        print(f"4. Import {tasks_file}")


if __name__ == "__main__":
    main()
