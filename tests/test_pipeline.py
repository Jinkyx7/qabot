from pathlib import Path

import pytest
from fpdf import FPDF

from qabot import document_loader, text_splitter


def _create_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Hello RAG PDF.\nThis is a small test document for chunking.")
    pdf.output(str(pdf_path))
    return pdf_path


def test_loader_and_splitter(tmp_path):
    pdf_path = _create_pdf(tmp_path)

    docs = document_loader(pdf_path)
    assert docs, "Loader should return at least one document"
    assert any("Hello RAG PDF" in doc.page_content for doc in docs)

    chunks = text_splitter(docs)
    assert chunks, "Splitter should return chunks"
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert "Hello RAG PDF" in chunks[0].page_content
