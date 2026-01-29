"""
Caption Matching Benchmark - Dataset Manifest

Provides data structures for benchmark dataset manifest (dataset.json).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentEntry:
    """Entry for a document in the benchmark dataset."""

    name: str
    annotation_path: str
    extraction_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "annotation_path": self.annotation_path,
        }
        if self.extraction_path:
            result["extraction_path"] = self.extraction_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentEntry":
        return cls(
            name=data["name"],
            annotation_path=data["annotation_path"],
            extraction_path=data.get("extraction_path"),
        )


@dataclass
class CaptionBenchmarkDataset:
    """Dataset manifest for caption matching benchmark."""

    name: str
    version: str
    annotator: str
    documents: List[DocumentEntry] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # Calculate statistics
        total_figures = 0
        total_tables = 0

        return {
            "name": self.name,
            "version": self.version,
            "annotator": self.annotator,
            "created_at": self.created_at,
            "statistics": {
                "total_documents": len(self.documents),
                "total_figures": total_figures,
                "total_tables": total_tables,
            },
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def load(cls, path: str) -> "CaptionBenchmarkDataset":
        """
        Load dataset from dataset.json file.

        Args:
            path: Path to the benchmark directory containing dataset.json

        Returns:
            CaptionBenchmarkDataset instance
        """
        dataset_path = Path(path)
        dataset_file = dataset_path / "dataset.json"

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            annotator=data.get("annotator", "unknown"),
            created_at=data.get("created_at", ""),
        )

        for doc_data in data.get("documents", []):
            dataset.documents.append(DocumentEntry.from_dict(doc_data))

        return dataset

    def save(self, path: str) -> None:
        """
        Save dataset to dataset.json file.

        Args:
            path: Path to the benchmark directory
        """
        dataset_path = Path(path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        dataset_file = dataset_path / "dataset.json"

        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_annotation_path(self, base_path: str, doc: DocumentEntry) -> Path:
        """Get absolute path to annotation file."""
        return Path(base_path) / doc.annotation_path

    def get_extraction_path(self, base_path: str, doc: DocumentEntry) -> Optional[Path]:
        """Get absolute path to extraction file."""
        if not doc.extraction_path:
            return None
        return Path(base_path) / doc.extraction_path
