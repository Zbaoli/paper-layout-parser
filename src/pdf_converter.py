"""
PDF to Image Converter Module

Converts PDF files to PNG images using PyMuPDF for document layout detection.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF


class PDFConverter:
    """Converts PDF files to images for layout detection."""

    def __init__(self, dpi: int = 200, output_format: str = "png"):
        """
        Initialize the PDF converter.

        Args:
            dpi: Resolution for image conversion (default: 200)
            output_format: Output image format (default: "png")
        """
        self.dpi = dpi
        self.output_format = output_format
        self.zoom = dpi / 72.0  # PDF default is 72 DPI

    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get information about a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)

        info = {
            "filename": pdf_path.name,
            "total_pages": len(doc),
            "metadata": doc.metadata,
            "page_sizes": [],
        }

        for page in doc:
            rect = page.rect
            info["page_sizes"].append({
                "width": rect.width,
                "height": rect.height,
            })

        doc.close()
        return info

    def convert_page(
        self,
        pdf_path: str,
        page_number: int,
        output_dir: str,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Convert a single PDF page to an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to convert (0-indexed)
            output_dir: Directory to save the output image

        Returns:
            Tuple of (output image path, (width, height))
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        page = doc[page_number]

        # Create transformation matrix for the desired DPI
        mat = fitz.Matrix(self.zoom, self.zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Generate output filename
        output_filename = f"page_{page_number + 1:04d}.{self.output_format}"
        output_path = output_dir / output_filename

        # Save the image
        pix.save(str(output_path))

        size = (pix.width, pix.height)

        doc.close()
        return str(output_path), size

    def convert_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        pages: Optional[List[int]] = None,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Convert all pages (or specified pages) of a PDF to images.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output images
            pages: List of page numbers to convert (0-indexed), None for all pages

        Returns:
            List of tuples containing (output path, (width, height)) for each page
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        # Create subdirectory named after the PDF
        pdf_name = pdf_path.stem
        pdf_output_dir = output_dir / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        if pages is None:
            pages = list(range(total_pages))

        results = []
        for page_num in pages:
            if 0 <= page_num < total_pages:
                output_path, size = self.convert_page(
                    pdf_path, page_num, pdf_output_dir
                )
                results.append((output_path, size))

        return results

    def convert_all_pdfs(
        self,
        input_dir: str,
        output_dir: str,
        progress_callback=None,
    ) -> dict:
        """
        Convert all PDF files in a directory.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save output images
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary mapping PDF names to their conversion results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        pdf_files = list(input_dir.glob("*.pdf"))
        results = {}

        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(i, len(pdf_files), pdf_path.name)

            try:
                conversion_result = self.convert_pdf(pdf_path, output_dir)
                results[pdf_path.stem] = {
                    "status": "success",
                    "pages": conversion_result,
                    "total_pages": len(conversion_result),
                }
            except Exception as e:
                results[pdf_path.stem] = {
                    "status": "error",
                    "error": str(e),
                }

        return results
