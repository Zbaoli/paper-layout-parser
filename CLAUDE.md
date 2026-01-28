# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocLayout-YOLO based PDF document layout detection system. Converts PDF pages to images and detects layout elements (titles, paragraphs, tables, images, formulas, etc.) using deep learning.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies (black, ruff)
uv sync --extra dev

# Run detection on all PDFs in data/papers/
uv run python main.py

# Process a single PDF
uv run python main.py --single-pdf data/papers/example.pdf

# Device selection
uv run python main.py --device mps        # Mac M-series GPU
uv run python main.py --device cuda       # NVIDIA GPU
uv run python main.py --device cpu        # CPU only

# Other options
uv run python main.py --confidence 0.5    # Custom confidence threshold
uv run python main.py --no-visualize      # Skip visualization output
uv run python main.py --extract           # Extract figures/tables with captions

# Code formatting and linting
uv run black src/ main.py --line-length 100
uv run ruff check src/ main.py
uv run ruff check src/ main.py --fix     # Auto-fix linting issues

# Caption matching benchmark
uv run python -m src.benchmark annotate --input data/output/paper1
uv run python -m src.benchmark annotate-batch --input data/output --vlm openai
uv run python -m src.benchmark build --input "data/output/*/caption_annotations.json" --output benchmark/caption-matching
uv run python -m src.benchmark evaluate --dataset benchmark/caption-matching
uv run python -m src.benchmark validate --dataset benchmark/caption-matching
uv run python -m src.benchmark report --inputs eval1.json eval2.json --output comparison.md
```

## Architecture

**Pipeline Flow**: PDF → Images (PyMuPDF) → Detection (DocLayout-YOLO) → JSON Results → Visualization → (Optional) Figure/Table Extraction

**Core Modules** (in `src/`):
- `pdf_converter.py` - PDFConverter: PDF to PNG conversion using PyMuPDF/fitz
- `layout_detector.py` - DocLayoutDetector with factory function `create_detector()` for instantiation
- `result_processor.py` - ResultProcessor: Structures detection results, calculates statistics, outputs JSON
- `visualizer.py` - Visualizer: Draws bounding boxes with class-specific colors and labels (BGR format)
- `figure_table_extractor.py` - FigureTableExtractor with CaptionMatcher: Crops figures/tables and matches captions by spatial proximity
- `benchmark.py` - Caption matching benchmark CLI: VLM annotation and evaluation
- `caption_matching/` - Caption matching evaluation module:
  - `evaluator.py` - CaptionMatchingEvaluator: Single-document evaluation against VLM ground truth
  - `benchmark.py` - CaptionMatchingBenchmark, BatchEvaluator: Batch evaluation on benchmark datasets
  - `reporter.py` - BenchmarkReporter: JSON and Markdown report generation
- `vlm_annotator/` - VLM-assisted annotation for ground truth generation

**Configuration**: `config/config.yaml` contains model settings, device preferences, path mappings, class definitions, and visualization colors (BGR format for OpenCV).

**Entry Point**: `main.py` orchestrates the pipeline with CLI argument handling.

## Detection Classes

DocLayout-YOLO (DocStructBench) detects 10 classes: Title, Plain-Text, Abandon, Figure, Figure-Caption, Table, Table-Caption, Table-Footnote, Isolate-Formula, Formula-Caption.

Class names are loaded dynamically from model metadata to ensure correct mapping.

## Output Structure

Each PDF is processed into its own directory under `data/output/`:

```
data/output/
├── legend.png                    # Class color legend
├── summary_report.json           # Batch processing summary (when processing multiple PDFs)
└── {pdf_name}/                   # One directory per PDF
    ├── pages/                    # Converted page images
    │   └── page_0001.png
    ├── annotated/                # Visualizations with bounding boxes
    │   └── page_0001.png
    ├── extractions/              # Extracted figures/tables (--extract)
    │   ├── figures/
    │   ├── tables/
    │   └── extraction_metadata.json
    └── result.json               # Detection results
```

## Caption Matching Benchmark

The caption matching benchmark evaluates figure/table-caption pairing accuracy against VLM-generated ground truth.

**Directory Structure:**
```
benchmark/
└── caption-matching/           # Caption matching benchmark
    ├── dataset.json            # Dataset manifest
    ├── annotations/            # Ground truth annotations (per document)
    │   └── {doc_name}/
    │       ├── caption_annotations.json
    │       └── extraction_metadata.json
    └── results/                # Evaluation results
        ├── eval_report.json
        └── eval_report.md
```

**Workflow:**
1. Process PDFs with `--extract` to generate detection results
2. Generate VLM annotations with `annotate-batch` or `annotate` command
3. Build benchmark dataset with `build`
4. Run evaluation with `evaluate`
