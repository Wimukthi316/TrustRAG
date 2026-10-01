"""Tests for the document-extraction plumbing behind the "attach a file" button.

These exercise `backend.app.services.extract` directly rather than through the
API, so they need no server and no model -- only pypdf and python-docx, both in
requirements.txt. The PDF is a tiny one-page fixture embedded as base64 so the
test carries its own input and depends on no PDF-writing package.
"""

from __future__ import annotations

import base64
from io import BytesIO

import pytest

from backend.app.services.extract import (
    MAX_BYTES,
    ExtractError,
    extract_text,
)

# One page, Helvetica, two lines, written with fpdf2 and verified to read back
# with pypdf. Kept as base64 so the test needs no PDF generator at run time.
_PDF_B64 = (
    "JVBERi0xLjMKJenr8b8KMSAwIG9iago8PAovQ291bnQgMQovS2lkcyBbMyAwIFJdCi9NZWRpYUJv"
    "eCBbMCAwIDU5NS4yOCA4NDEuODldCi9UeXBlIC9QYWdlcwo+PgplbmRvYmoKMiAwIG9iago8PAov"
    "T3BlbkFjdGlvbiBbMyAwIFIgL0ZpdEggbnVsbF0KL1BhZ2VMYXlvdXQgL09uZUNvbHVtbgovUGFn"
    "ZXMgMSAwIFIKL1R5cGUgL0NhdGFsb2cKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDQg"
    "MCBSCi9QYXJlbnQgMSAwIFIKL1Jlc291cmNlcyA2IDAgUgovVHlwZSAvUGFnZQo+PgplbmRvYmoK"
    "NCAwIG9iago8PAovRmlsdGVyIC9GbGF0ZURlY29kZQovTGVuZ3RoIDE1NAo+PgpzdHJlYW0KeJw9"
    "zEELgjAAhuG7v+I71mU5RadHIw2TIGyXTjJt5kqmuEX07zOJ4D29h8fDwXFJwPBythybjIJ6xHX"
    "BW6T8u3xKaAQWRySc7xWrc77Py/ySVKddVh2TskhLKAOh5xoltcU0NA+0w2QnaQxZg99/VhDGhI"
    "VgLCCxv2CF0jcUwnRvMQrUT9VbdLM2il40EoOG7eQC/p0PJ1sx7AplbmRzdHJlYW0KZW5kb2Jq"
    "CjUgMCBvYmoKPDwKL0Jhc2VGb250IC9IZWx2ZXRpY2EKL0VuY29kaW5nIC9XaW5BbnNpRW5jb2Rp"
    "bmcKL1N1YnR5cGUgL1R5cGUxCi9UeXBlIC9Gb250Cj4+CmVuZG9iago2IDAgb2JqCjw8Ci9Gb250"
    "IDw8L0YxIDUgMCBSPj4KL1Byb2NTZXQgWy9QREYgL1RleHQgL0ltYWdlQiAvSW1hZ2VDIC9JbWFn"
    "ZUldCj4+CmVuZG9iago3IDAgb2JqCjw8Ci9DcmVhdGlvbkRhdGUgKEQ6MjAyNjA5MzAxOTIzMjla"
    "KQo+PgplbmRvYmoKeHJlZgowIDgKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAwMDAw"
    "IG4gCjAwMDAwMDAxMDIgMDAwMDAgbiAKMDAwMDAwMDIwNSAwMDAwMCBuIAowMDAwMDAwMjg1IDAw"
    "MDAwIG4gCjAwMDAwMDA1MTEgMDAwMDAgbiAKMDAwMDAwMDYwOCAwMDAwMCBuIAowMDAwMDAwNjk1"
    "IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgOAovUm9vdCAyIDAgUgovSW5mbyA3IDAgUgovSUQg"
    "WzwyMTlGNUZFMEI5NDVCMDIyQkE3MDUzMENGNjgzRjk3RT48MjE5RjVGRTBCOTQ1QjAyMkJBNzA1"
    "MzBDRjY4M0Y5N0U+XQo+PgpzdGFydHhyZWYKNzUwCiUlRU9GCg=="
)


def _docx_bytes(with_table: bool = False) -> bytes:
    import docx

    document = docx.Document()
    document.add_paragraph("Sigiriya is an ancient rock fortress in Sri Lanka.")
    document.add_paragraph("King Kashyapa selected the site as his new capital.")
    if with_table:
        table = document.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Year"
        table.rows[0].cells[1].text = "Event"
        table.rows[1].cells[0].text = "477"
        table.rows[1].cells[1].text = "Capital built"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_pdf_returns_text_and_page_count():
    result = extract_text("source.pdf", base64.b64decode(_PDF_B64))
    assert "SIGIRIYA_PDF_MARKER" in result.text
    assert "Kashyapa" in result.text
    assert result.pages == 1
    assert result.note is None


def test_docx_paragraphs():
    result = extract_text("source.docx", _docx_bytes())
    assert "Sigiriya" in result.text
    assert "Kashyapa" in result.text
    assert result.pages is None
    assert result.note is None


def test_docx_reads_table_cells():
    result = extract_text("source.docx", _docx_bytes(with_table=True))
    assert "477 | Capital built" in result.text


def test_txt_roundtrip():
    result = extract_text("notes.txt", b"Sigiriya sits on a rock.\nKashyapa built it.")
    assert "Sigiriya sits on a rock." in result.text
    assert "Kashyapa built it." in result.text


def test_markdown_normalises_newlines_and_collapses_blanks():
    result = extract_text("notes.md", b"# Title\r\n\r\n\r\nbody text\r\n")
    assert "\r" not in result.text
    assert result.text == "# Title\n\nbody text"


def test_non_utf8_text_falls_back_to_cp1252():
    result = extract_text("latin.txt", "café résumé".encode("cp1252"))
    assert "café résumé" in result.text


def test_empty_file_rejected():
    with pytest.raises(ExtractError, match="empty"):
        extract_text("a.txt", b"")


def test_oversized_file_rejected():
    with pytest.raises(ExtractError, match="limit"):
        extract_text("a.txt", b"x" * (MAX_BYTES + 1))


def test_unsupported_extension_rejected():
    with pytest.raises(ExtractError, match="not supported"):
        extract_text("a.exe", b"stuff")


def test_missing_extension_rejected():
    with pytest.raises(ExtractError, match="not supported"):
        extract_text("README", b"stuff")


def test_corrupt_pdf_rejected():
    with pytest.raises(ExtractError):
        extract_text("a.pdf", b"not really a pdf")


def test_corrupt_docx_rejected():
    with pytest.raises(ExtractError):
        extract_text("a.docx", b"not a zip at all")
