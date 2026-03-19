"""
Tests for nima.extractor

We test the pure helper functions (no PDF file needed) and mock
pdfplumber for the integration-level extract() test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nima.extractor import (
    classify_item,
    extract_code,
    parse_greek_amount,
    ScannedPDFError,
    extract,
)
from nima.models import DocumentType, ItemType


# ---------------------------------------------------------------------------
# parse_greek_amount
# ---------------------------------------------------------------------------

class TestParseGreekAmount:
    def test_simple_integer(self):
        assert parse_greek_amount("1.000") == 1000.0

    def test_decimal(self):
        assert parse_greek_amount("1.234,56") == 1234.56

    def test_large_amount(self):
        assert parse_greek_amount("1.234.567,89") == 1234567.89

    def test_no_thousands(self):
        assert parse_greek_amount("500,00") == 500.0

    def test_plain_integer_string(self):
        assert parse_greek_amount("42") == 42.0

    def test_empty_string(self):
        assert parse_greek_amount("") is None

    def test_non_numeric(self):
        assert parse_greek_amount("N/A") is None

    def test_with_euro_symbol(self):
        # Some PDFs include the € symbol
        assert parse_greek_amount("€ 1.000,00") == 1000.0


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_simple_two_digit(self):
        code, desc = extract_code("02 Λειτουργικές δαπάνες")
        assert code == "02"
        assert "Λειτουργικές" in desc

    def test_dotted_code(self):
        code, desc = extract_code("02.00 Μισθοδοσία")
        assert code == "02.00"

    def test_no_code(self):
        code, desc = extract_code("Σύνολο εσόδων")
        assert code == ""
        assert desc == "Σύνολο εσόδων"

    def test_long_code(self):
        code, desc = extract_code("64-7135.001 Κατασκευή πεζοδρομίων")
        assert "64" in code
        assert "Κατασκευή" in desc


# ---------------------------------------------------------------------------
# classify_item
# ---------------------------------------------------------------------------

class TestClassifyItem:
    def test_revenue_keyword(self):
        assert classify_item("Έσοδα από τέλη", DocumentType.BUDGET) == ItemType.REVENUE

    def test_technical_doc_overrides(self):
        # Any item in a technical programme is TECHNICAL regardless of text
        assert classify_item("Μισθοδοσία", DocumentType.TECHNICAL_PROGRAMME) == ItemType.TECHNICAL

    def test_default_expense(self):
        assert classify_item("Γενικές δαπάνες", DocumentType.BUDGET) == ItemType.EXPENSE

    def test_construction_keyword(self):
        result = classify_item("Κατασκευή νέου δρόμου", DocumentType.BUDGET)
        assert result == ItemType.TECHNICAL


# ---------------------------------------------------------------------------
# extract() — integration test with mocked pdfplumber
# ---------------------------------------------------------------------------

def _make_mock_page(tables=None, text=""):
    page = MagicMock()
    page.page_number = 1
    page.extract_tables.return_value = tables or []
    page.extract_text.return_value = text
    return page


class TestExtract:
    def test_extracts_from_table(self, tmp_path):
        """extract() should parse a simple two-column table correctly."""
        fake_table = [
            ["02 Μισθοδοσία προσωπικού", "150.000,00"],
            ["10 Λοιπές δαπάνες", "50.000,00"],
        ]
        mock_page = _make_mock_page(tables=[fake_table])

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            pdf_file = tmp_path / "budget.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake")

            result = extract(pdf_file, document_id=1)

        assert len(result.items) == 2
        assert result.items[0].amount == 150_000.0
        assert result.items[1].amount == 50_000.0
        assert result.items[0].code == "02"

    def test_scanned_pdf_raises(self, tmp_path):
        """extract() should raise ScannedPDFError when no text is found."""
        mock_page = _make_mock_page(tables=[], text="")
        mock_page.extract_text.return_value = ""

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            pdf_file = tmp_path / "scanned.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake")

            with pytest.raises(ScannedPDFError):
                extract(pdf_file, document_id=1)

    def test_warnings_for_bad_amounts(self, tmp_path):
        """extract() should record a warning for unparseable amounts."""
        fake_table = [
            ["02 Μισθοδοσία", "N/A"],
        ]
        mock_page = _make_mock_page(tables=[fake_table])

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = [mock_page]
            mock_open.return_value = mock_pdf

            pdf_file = tmp_path / "budget.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake")

            result = extract(pdf_file, document_id=1)

        assert any("N/A" in w for w in result.warnings)
        assert result.items[0].amount is None