"""
Caption Matching Benchmark - Dataset Builder

Builds benchmark datasets from existing annotation files.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from .manifest import CaptionBenchmarkDataset, DocumentEntry


class DatasetBuilder:
    """Builds benchmark dataset from existing annotation files."""

    def __init__(self, name: str = "caption-matching-v1", version: str = "1.0.0"):
        """
        Initialize the dataset builder.

        Args:
            name: Dataset name
            version: Dataset version
        """
        self.name = name
        self.version = version

    def build_from_annotations(
        self,
        annotation_paths: List[str],
        output_dir: str,
        copy_files: bool = True,
    ) -> CaptionBenchmarkDataset:
        """
        Build dataset from annotation files.

        Args:
            annotation_paths: List of paths to caption_annotations.json files
            output_dir: Output directory for the benchmark dataset
            copy_files: Whether to copy annotation files to output directory

        Returns:
            CaptionBenchmarkDataset instance
        """
        output_path = Path(output_dir)
        annotations_dir = output_path / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        documents = []
        annotator = "unknown"

        for ann_path in annotation_paths:
            ann_file = Path(ann_path)
            if not ann_file.exists():
                print(f"Warning: Annotation file not found: {ann_file}")
                continue

            # Load annotation to get PDF name
            with open(ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            pdf_name = ann_data.get("pdf_name", ann_file.parent.name)
            if annotator == "unknown":
                annotator = ann_data.get("annotator", "unknown")

            # Create document directory
            doc_dir = annotations_dir / pdf_name
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Copy or reference annotation file
            if copy_files:
                dest_ann = doc_dir / "caption_annotations.json"
                shutil.copy(ann_file, dest_ann)
                ann_rel_path = f"annotations/{pdf_name}/caption_annotations.json"
            else:
                ann_rel_path = str(ann_file)

            # Try to find extraction_metadata.json
            extraction_path = ann_file.parent / "extractions" / "extraction_metadata.json"
            ext_rel_path = None
            if extraction_path.exists():
                if copy_files:
                    dest_ext = doc_dir / "extraction_metadata.json"
                    shutil.copy(extraction_path, dest_ext)
                    ext_rel_path = f"annotations/{pdf_name}/extraction_metadata.json"
                else:
                    ext_rel_path = str(extraction_path)

            documents.append(
                DocumentEntry(
                    name=pdf_name,
                    annotation_path=ann_rel_path,
                    extraction_path=ext_rel_path,
                )
            )

        dataset = CaptionBenchmarkDataset(
            name=self.name,
            version=self.version,
            annotator=annotator,
            documents=documents,
            created_at=datetime.now().isoformat(),
        )

        # Save dataset manifest
        dataset.save(str(output_path))

        return dataset

    def build_from_output_dir(
        self,
        output_dir: str,
        benchmark_dir: str,
    ) -> CaptionBenchmarkDataset:
        """
        Build dataset from data/output directory structure.

        Args:
            output_dir: Path to data/output directory
            benchmark_dir: Output directory for benchmark dataset

        Returns:
            CaptionBenchmarkDataset instance
        """
        output_path = Path(output_dir)

        # Find all caption_annotations.json files
        annotation_paths = []
        for ann_file in output_path.glob("*/caption_annotations.json"):
            annotation_paths.append(str(ann_file))

        if not annotation_paths:
            raise ValueError(f"No caption_annotations.json files found in {output_path}")

        return self.build_from_annotations(
            annotation_paths=annotation_paths,
            output_dir=benchmark_dir,
            copy_files=True,
        )
