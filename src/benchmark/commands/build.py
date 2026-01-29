"""
Build Command

Build benchmark dataset from annotation files.
"""

import glob
import sys
from pathlib import Path
from typing import List


def register_parser(subparsers):
    """Register build command parser."""
    build_parser = subparsers.add_parser("build", help="Build caption matching benchmark dataset")
    build_parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Paths to caption_annotations.json files or glob patterns",
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default="benchmark/caption-matching",
        help="Output directory for benchmark dataset",
    )
    build_parser.add_argument(
        "--name",
        type=str,
        default="caption-matching-v1",
        help="Dataset name",
    )
    build_parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version",
    )


def run(args):
    """Run dataset build command."""
    from ..batch_evaluator import DatasetBuilder

    # Expand glob patterns
    annotation_files = []
    for pattern in args.input:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            annotation_files.extend(matches)
        elif Path(pattern).exists():
            annotation_files.append(pattern)
        else:
            print(f"Warning: No files found for pattern: {pattern}")

    if not annotation_files:
        print("Error: No annotation files found")
        sys.exit(1)

    print(f"Building benchmark dataset from {len(annotation_files)} annotation files...")

    builder = DatasetBuilder(name=args.name, version=args.version)
    dataset = builder.build_from_annotations(
        annotation_paths=annotation_files,
        output_dir=args.output,
        copy_files=True,
    )

    print("\n" + "=" * 50)
    print("Benchmark Dataset Created")
    print("=" * 50)
    print(f"Name: {dataset.name}")
    print(f"Version: {dataset.version}")
    print(f"Documents: {len(dataset.documents)}")
    print(f"Output: {args.output}")
    print(f"\nDataset manifest: {args.output}/dataset.json")
