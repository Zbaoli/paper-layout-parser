# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocLayout-YOLO based PDF document layout detection system. Converts PDF pages to images and detects layout elements (titles, paragraphs, tables, images, formulas, etc.) using deep learning.

## Commands

```bash
# Install dependencies
uv sync
uv sync --extra dev     # With dev dependencies (black, ruff, pytest)
uv sync --extra vlm     # With VLM dependencies (for benchmark annotation)

# Run detection
uv run python main.py                                    # All PDFs in data/papers/
uv run python main.py --single-pdf data/papers/test.pdf # Single PDF
uv run python main.py --device cuda                     # NVIDIA GPU
uv run python main.py --device mps                      # Mac M-series
uv run python main.py --extract                         # Extract figures/tables with captions

# Tests
uv run pytest tests/ -v                                 # All tests
uv run pytest tests/test_matching/ -v                   # Single test module
uv run pytest tests/test_matching/test_caption_matcher.py::TestHorizontalOverlap -v  # Single class

# Linting
uv run black src/ benchmarks/ main.py --line-length 100
uv run ruff check src/ benchmarks/ main.py --fix

# Benchmark CLI
uv run python -m benchmarks.caption_matching --help
uv run python -m benchmarks.caption_matching annotate --input data/output/paper1
uv run python -m benchmarks.caption_matching evaluate
```

## Architecture

**Pipeline**: PDF → Images (PyMuPDF) → Detection (DocLayout-YOLO) → JSON → Visualization → Figure/Table Extraction

```
src/doclayout/           # Core package (pip installable)
├── core/                # PDF conversion, detection, result processing
├── matching/            # CaptionMatcher: spatial proximity matching
└── visualization/       # BoundingBoxRenderer with strategy pattern

benchmarks/              # Evaluation tools (not packaged)
├── caption_matching/    # CLI + evaluator for caption matching accuracy
└── tools/vlm_annotator/ # VLM-assisted ground truth annotation
```

**Key Classes**:
- `PDFConverter`: PDF to PNG using PyMuPDF
- `DocLayoutDetector`: YOLO inference wrapper with `create_detector()` factory
- `CaptionMatcher`: Matches figures/tables to captions by spatial proximity
- `BoundingBoxRenderer`: Visualization with pluggable `LabelStrategy`

## Public API

```python
from doclayout import (
    PDFConverter, create_detector, DocLayoutDetector, Detection,
    ResultProcessor, FigureTableExtractor,
    CaptionMatcher, SearchDirection, ExtractedItem, ExtractionResult,
    BoundingBoxRenderer, create_visualizer, ColorPalette,
)
```

## Detection Classes

DocLayout-YOLO detects 10 classes: Title, Plain-Text, Abandon, Figure, Figure-Caption, Table, Table-Caption, Table-Footnote, Isolate-Formula, Formula-Caption.

## Data Directories

- `data/papers/` - Input PDFs
- `data/output/{pdf_name}/` - Detection results per PDF (pages/, annotated/, result.json)
- `data/benchmark/` - Benchmark datasets

## Configuration

`config/config.yaml` contains model settings, device preferences, and visualization colors (BGR format for OpenCV).
