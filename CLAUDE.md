# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv8-based PDF document layout detection system. Converts PDF pages to images and detects layout elements (titles, paragraphs, tables, images, formulas, etc.) using deep learning models.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies (pytest, black, ruff)
uv sync --extra dev

# Run detection on all PDFs in data/papers/
uv run python main.py

# Process a single PDF
uv run python main.py --single-pdf data/papers/example.pdf

# Switch models
uv run python main.py --model doclayout   # DocLayout-YOLO (default)
uv run python main.py --model yolov8      # YOLOv8-DocLayNet

# Device selection
uv run python main.py --device mps        # Mac M-series GPU
uv run python main.py --device cpu        # CPU only

# Other options
uv run python main.py --confidence 0.5    # Custom confidence threshold
uv run python main.py --no-visualize      # Skip visualization output

# Code formatting and linting
uv run black src/ main.py --line-length 100
uv run ruff check src/ main.py
```

## Architecture

**Pipeline Flow**: PDF → Images (PyMuPDF) → Detection (YOLO) → JSON Results → Visualization

**Core Modules** (in `src/`):
- `pdf_converter.py` - PDFConverter: PDF to PNG conversion using PyMuPDF/fitz
- `layout_detector.py` - Abstract BaseLayoutDetector with DocLayoutDetector and YOLOv8LayoutDetector implementations. Factory function `create_detector()` for instantiation
- `result_processor.py` - ResultProcessor: Structures detection results, calculates statistics, outputs JSON
- `visualizer.py` - Visualizer: Draws bounding boxes with class-specific colors and labels

**Configuration**: `config/config.yaml` contains model settings, device preferences, path mappings, class definitions, and visualization colors (BGR format for OpenCV).

**Entry Point**: `main.py` orchestrates the pipeline with CLI argument handling.

## Detection Classes

DocLayout-YOLO detects 10 classes; YOLOv8-DocLayNet detects 11 classes (adds Footnote). Class names are loaded dynamically from model metadata to ensure correct mapping.

## Output Structure

- `data/images/` - Converted page images (organized by PDF name)
- `data/results/json/` - Detection results with per-page and per-document statistics
- `data/results/visualizations/` - Annotated images with bounding boxes
