"""
nima.extractor
--------------
Converts a municipal PDF file into a list of structured BudgetItemCreate
objects ready to be persisted to the database.

Extraction strategy
~~~~~~~~~~~~~~~~~~~
Greek municipal PDFs arrive in two main structural formats:

1. TABLE-BASED (primary strategy)
   Most PDFs embed proper table bounding boxes. pdfplumber detects these
   and returns a list-of-lists structure. We scan right-to-left for the
   amount column because some PDFs put totals in column 2, others in the
   last column.

2. LINE-BASED (fallback)
   Some PDFs render table-like layouts using positioned text without any
   embedded table structure. We fall back to regex-based line parsing,
   looking for lines where a description is followed by two or more spaces
   and then a number.

3. SCANNED / IMAGE-ONLY (not handled)
   If pdfplumber finds no text at all, we raise ScannedPDFError with a
   clear message pointing to ocrmypdf as the recommended pre-processor.
   Silently returning zero items would be worse than failing loudly.

Confidence scoring
~~~~~~~~~~~~~~~~~~
Every item receives a confidence_score (0.0–1.0) that reflects how
reliable the extraction was. The API exposes this so the dashboard can
flag uncertain data to citizens rather than presenting it as ground truth.

Civic labelling
~~~~~~~~~~~~~~~
Raw Greek bureaucratic headings (e.g. "ΥΠΗΡΕΣΙΑ 30") are mapped to
plain-English labels (e.g. "Public Works & Infrastructure"). These labels
are stored in BudgetItem.subcategory and used by the dashboard's
citizen-friendly view.

Known limitations
~~~~~~~~~~~~~~~~~
- Amounts using a space as a thousands separator (e.g. "1 000,00") are
  not parsed. Extend _AMOUNT_RE if this format is encountered.
- Multi-page tables that span a page break may be split into two items.
- KAE code detection is heuristic; unusual formats may be missed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

from nima.models import BudgetItemCreate, DocumentType, ItemType


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Base class for all extraction failures."""


class ScannedPDFError(ExtractionError):
    """
    Raised when the PDF contains no text layer (image-only / scanned).

    This is a common issue with older municipal documents. The recommended
    fix is to run the file through ocrmypdf before ingesting:
        ocrmypdf input.pdf output.pdf && nima ingest output.pdf ...
    """


# ---------------------------------------------------------------------------
# Civic category mapping
# ---------------------------------------------------------------------------
# Maps substrings of raw Greek category headings to plain-English labels.
# Keys are lowercase; matching is substring-based (see civic_label()).
# Add entries here as new PDF formats are encountered — do not hardcode
# category strings anywhere else in the codebase.

CIVIC_CATEGORY_MAP: dict[str, str] = {
    "υπηρεσια 20":        "Sanitation & Waste",
    "υπηρεσια 30":        "Public Works & Infrastructure",
    "υπηρεσια 35":        "Green Spaces & Environment",
    "υπηρεσια 40":        "Urban Planning",
    "υπηρεσια 45":        "Cemeteries",
    "υπηρεσια 55":        "Social Services",
    "υπηρεσια 64":        "School Buildings",
    "υπηρεσια 70":        "Waterfront & Special Projects",
    "μισθοδοσια":         "Staff Salaries",
    "αμοιβες":            "Professional Fees",
    "λειτουργικες":       "Operating Costs",
    "προμηθειες":         "Supplies & Materials",
    "κοινωνικες":         "Social Services",
    "τεχνικες":           "Technical Works",
    "γενικες υπηρεσιες":  "General Administration",
    "καθαριοτητα":        "Sanitation & Waste",
    "πρασινο":            "Green Spaces & Environment",
    "σχολικη στεγη":      "School Buildings",
    "αναπλασεις":         "Urban Renewal",
    "οδικο δικτυο":       "Road Network",
    "περιβαλλον":         "Environment & Climate",
    "κοινωνικες υποδομες":"Social Infrastructure",
    "αστικος σχεδιασμος": "Urban Planning",
    "τακτικα εσοδα":      "Regular Revenue",
    "εκτακτα εσοδα":      "Extraordinary Revenue",
    "εσοδα":              "Revenue",
    "δαπανες":            "Expenditure",
}

# Maps the first two digits of a KAE budget code to a civic label.
# Used as a secondary fallback when the category heading is not recognised.
KAE_PREFIX_MAP: dict[str, str] = {
    "00": "General Administration",
    "01": "Tax Revenue",
    "02": "Fees & Charges",
    "10": "Extraordinary Revenue",
    "11": "EU & State Grants",
    "20": "Sanitation & Waste",
    "30": "Public Works",
    "35": "Green Spaces",
    "40": "Urban Planning",
    "45": "Cemeteries",
    "55": "Social Services",
    "60": "Education",
    "64": "School Buildings",
    "70": "Special Projects",
}


def civic_label(category: Optional[str], code: Optional[str]) -> Optional[str]:
    """
    Return a plain-English civic label for a budget item.

    Tries CIVIC_CATEGORY_MAP first (substring match on the raw category
    heading), then falls back to KAE_PREFIX_MAP using the first two digits
    of the budget code. Returns None if neither matches.
    """
    if category:
        lower = category.lower().strip()
        for key, label in CIVIC_CATEGORY_MAP.items():
            if key in lower:
                return label
    if code:
        prefix = code[:2]
        if prefix in KAE_PREFIX_MAP:
            return KAE_PREFIX_MAP[prefix]
    return None


# ---------------------------------------------------------------------------
# Noise / header row filters
# ---------------------------------------------------------------------------
# These patterns match rows that look like data but are actually page
# headers, footers, or summary totals that should not become BudgetItems.

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*σελίδα\s+\d+", re.IGNORECASE),   # "Σελίδα 1"
    re.compile(r"^\s*page\s+\d+",   re.IGNORECASE),   # "Page 1"
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$"),     # date stamps
    re.compile(r"^\s*σύνολο\s*$",   re.IGNORECASE),   # bare "Σύνολο"
    re.compile(r"^\s*περιγραφή\s*$",re.IGNORECASE),   # column header
    re.compile(r"^\s*α/α\s*$",      re.IGNORECASE),   # sequence number header
    re.compile(r"^\s*\d{1,3}\s*$"),                   # lone sequence numbers
]

# Summary rows contain totals that are already the sum of individual items.
# Including them would double-count spending in the dashboard.
_SUMMARY_KEYWORDS: set[str] = {
    "γενικο συνολο",
    "συνολο υπηρεσι",
    "grand total",
    "συνολικο τεχνικο",
    "γενικο συνολο τεχνικου",
}


def _is_noise(description: str) -> bool:
    """
    Return True if this description string should be discarded.

    Called on every extracted row before creating a BudgetItem.
    """
    stripped = description.strip()
    if not stripped:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.match(stripped):
            return True
    lower = stripped.lower()
    return any(kw in lower for kw in _SUMMARY_KEYWORDS)


# ---------------------------------------------------------------------------
# Internal row representation
# ---------------------------------------------------------------------------

@dataclass
class RawRow:
    """
    Intermediate, unvalidated data from a single PDF row.

    This exists between raw extraction and BudgetItemCreate creation.
    Having an explicit intermediate type makes it easier to add
    validation or enrichment steps without changing the extraction logic.
    """
    description: str
    amount_str:  str           # raw string before numeric parsing
    code:        Optional[str]
    source_page: int
    raw_text:    str           # verbatim cell content for audit trail
    category:    Optional[str] = None
    subcategory: Optional[str] = None


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------

# Greek number format: thousands separator = ".", decimal separator = ","
# e.g.  "1.234.567,89"  ->  1234567.89
_AMOUNT_RE = re.compile(r"[\d.,]+")


def parse_greek_amount(text: str) -> Optional[float]:
    """
    Parse a number string into a Python float.

    Handles two formats found in Greek municipal PDFs:

    Greek format  — thousands separator = ".", decimal separator = ","
      e.g.  "1.234.567,89"  ->  1234567.89

    English format — thousands separator = ",", decimal separator = "."
      e.g.  "1,234,567.89"  ->  1234567.89

    Detection heuristic: if the string ends with a comma followed by
    exactly two digits, treat it as Greek decimal format.
    Otherwise treat it as English format.
    """
    # Strip currency symbols before matching
    cleaned = text.strip().replace("EUR", "").replace("\u20ac", "").strip()
    match = _AMOUNT_RE.search(cleaned)
    if not match:
        return None
    raw = match.group()
    try:
        if re.search(r",\d{2}$", raw):   # Greek format: "1.234,56"
            normalised = raw.replace(".", "").replace(",", ".")
        else:                              # English format: "1,234.56"
            normalised = raw.replace(",", "")
        val = float(normalised)
        # Discard near-zero values — usually placeholder cells
        return val if val >= 0.01 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# KAE / KA code extraction
# ---------------------------------------------------------------------------

# Budget codes: two digits optionally followed by more digits, dots, or dashes
# e.g.  "02", "02.00", "64-7135.001"
_CODE_RE = re.compile(r"^\s*(\d{2}[\d.\-]{0,12})\s+")


def extract_code(cell: str) -> tuple[str, str]:
    """
    Split a description cell into (budget_code, description_text).

    Returns ("", original_cell) if no code prefix is found.
    """
    match = _CODE_RE.match(cell)
    if match:
        code        = match.group(1).strip()
        description = cell[match.end():].strip()
        return code, description
    return "", cell.strip()


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    amount:      Optional[float],
    code:        Optional[str],
    description: str,
    amount_str:  str,
) -> float:
    """
    Return a confidence score (0.0–1.0) for a single extracted item.

    Scoring logic:
      - No parseable amount            -> 0.1 (almost certainly a header row)
      - Valid amount                   -> 0.7 base
      - + KAE code present             -> +0.15 (structured, machine-readable)
      - + Description longer than 10ch -> +0.05 (enough context to verify)
      - - Amount is a round 1000s      -> -0.10 (might be a budget estimate)
      - - Amount is very small (<100)  -> -0.20 (likely a noise row)

    The result is clamped to [0.0, 1.0] and rounded to two decimal places.
    """
    if amount is None:
        return 0.1

    score = 0.7

    if code:
        score += 0.15                    # structured KAE code increases trust

    if len(description) > 10:
        score += 0.05                    # meaningful description

    if amount > 0 and amount % 1000 == 0:
        score -= 0.10                    # suspiciously round figure

    if amount < 100:
        score -= 0.20                    # very small amounts are often noise

    return round(max(0.0, min(1.0, score)), 2)


# ---------------------------------------------------------------------------
# Item type classification
# ---------------------------------------------------------------------------

# Keyword sets use accented-stripped lowercase for robustness.
# pdfplumber sometimes returns accented characters inconsistently across PDFs.
_REVENUE_KEYWORDS: set[str] = {
    "εσοδ", "εισπραξ", "επιχορηγηση", "τελη",
    "φορο", "εισφορα", "τοκοι", "δωρεα",
}
_TECHNICAL_KEYWORDS: set[str] = {
    "εργο", "κατασκευη", "ανακαινιση", "συντηρηση",
    "προμηθεια", "μελετη", "εκπονηση", "αναπλαση",
    "ανεγερση", "αποκατασταση", "ενισχυση",
}


def classify_item(description: str, doc_type: DocumentType) -> ItemType:
    """
    Classify a budget item as REVENUE, EXPENSE, or TECHNICAL.

    Documents of type TECHNICAL_PROGRAMME always produce TECHNICAL items —
    no keyword matching is needed. For BUDGET documents, we scan the
    description for revenue or technical keywords and fall back to EXPENSE.
    """
    # All items in a technical programme are capital projects by definition
    if doc_type == DocumentType.TECHNICAL_PROGRAMME:
        return ItemType.TECHNICAL

    lower = description.lower()
    if any(k in lower for k in _REVENUE_KEYWORDS):
        return ItemType.REVENUE
    if any(k in lower for k in _TECHNICAL_KEYWORDS):
        return ItemType.TECHNICAL
    return ItemType.EXPENSE


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def _rows_from_table(
    table:            list[list[Optional[str]]],
    page_num:         int,
    current_category: Optional[str],
) -> tuple[list[RawRow], Optional[str]]:
    """
    Extract RawRows from a pdfplumber table structure.

    Returns (rows, updated_category) so the caller can thread the
    current category heading across multiple tables on the same page.

    Column detection heuristic:
      - First non-empty cell  -> description
      - Last non-empty cell   -> amount (right-to-left scan as fallback)
      - Rows with only one non-empty cell are treated as category headings
    """
    rows: list[RawRow] = []

    for cells in table:
        # Skip completely empty rows
        if not cells or all(c is None or c.strip() == "" for c in cells):
            continue

        cells    = [c or "" for c in cells]
        raw_text = " | ".join(cells)
        non_empty = [c.strip() for c in cells if c.strip()]

        # Single-cell rows are category headings, not data rows
        if len(non_empty) == 1:
            current_category = non_empty[0]
            continue

        description_cell = cells[0].strip()
        amount_cell      = cells[-1].strip()

        if not description_cell or _is_noise(description_cell):
            continue

        # If the last cell is not numeric, scan right-to-left for one that is.
        # Some PDF layouts put notes or dates in the last column.
        if not _AMOUNT_RE.search(amount_cell):
            for cell in reversed(cells[1:]):
                if _AMOUNT_RE.search(cell.strip()):
                    amount_cell = cell.strip()
                    break

        code, description = extract_code(description_cell)

        rows.append(RawRow(
            description = description or description_cell,
            amount_str  = amount_cell,
            code        = code or None,
            source_page = page_num,
            raw_text    = raw_text,
            category    = current_category,
        ))

    return rows, current_category


# A "data line" must end with a number preceded by two or more spaces.
# Single-space separation is too common in Greek text to be reliable.
_LINE_WITH_AMOUNT = re.compile(r"^(.+?)\s{2,}([\d.,]{3,})\s*$")


def _rows_from_text(
    text:             str,
    page_num:         int,
    current_category: Optional[str],
) -> tuple[list[RawRow], Optional[str]]:
    """
    Fallback extraction: parse raw text lines looking for description + amount.

    Used when pdfplumber finds no embedded table structure on a page.
    Lines that contain only text (no digits) are treated as category headings
    if they are longer than 4 characters.
    """
    rows: list[RawRow] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or _is_noise(line):
            continue

        match = _LINE_WITH_AMOUNT.match(line)
        if not match:
            # No amount found — treat as a potential category heading
            if len(line) > 4 and not any(ch.isdigit() for ch in line):
                current_category = line
            continue

        description_cell = match.group(1).strip()
        amount_str       = match.group(2)

        if _is_noise(description_cell):
            continue

        code, description = extract_code(description_cell)

        rows.append(RawRow(
            description = description or description_cell,
            amount_str  = amount_str,
            code        = code or None,
            source_page = page_num,
            raw_text    = line,
            category    = current_category,
        ))

    return rows, current_category


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    The result of extracting one PDF file.

    items              — BudgetItemCreate objects ready for database insertion
    page_count         — total pages in the source PDF
    warnings           — non-fatal issues (unparseable amounts, blank pages)
    items_with_amount  — items where a numeric amount was successfully parsed
    items_without_amount — items where amount parsing failed
    avg_confidence     — mean confidence score across all items (0.0–1.0)
    """
    items:                 list[BudgetItemCreate]
    page_count:            int
    warnings:              list[str] = field(default_factory=list)
    items_with_amount:     int       = 0
    items_without_amount:  int       = 0
    avg_confidence:        float     = 0.0


def extract(
    pdf_path:    Path,
    document_id: int,
    doc_type:    DocumentType = DocumentType.BUDGET,
) -> ExtractionResult:
    """
    Extract budget items from a municipal PDF file.

    Parameters
    ----------
    pdf_path    : Path to the PDF file. Must exist and be text-layer PDF.
    document_id : The id of the Document record already inserted in the DB.
                  All extracted items are linked to this document.
    doc_type    : Whether this is a budget or technical programme PDF.
                  Affects item type classification.

    Returns
    -------
    ExtractionResult with all extracted items and quality metrics.

    Raises
    ------
    FileNotFoundError  — pdf_path does not exist
    ScannedPDFError    — no text layer detected across the whole document
    ExtractionError    — any other unrecoverable failure
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    items:            list[BudgetItemCreate] = []
    warnings:         list[str]             = []
    confidences:      list[float]           = []
    total_text_chars: int                   = 0
    current_category: Optional[str]         = None

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        for page in pdf.pages:
            page_num  = page.page_number  # 1-indexed
            raw_rows: list[RawRow] = []

            # --- Primary strategy: embedded table detection ---
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    new_rows, current_category = _rows_from_table(
                        table, page_num, current_category
                    )
                    raw_rows.extend(new_rows)

            else:
                # --- Fallback strategy: line-based text parsing ---
                text = page.extract_text() or ""
                total_text_chars += len(text)

                if text.strip():
                    new_rows, current_category = _rows_from_text(
                        text, page_num, current_category
                    )
                    raw_rows.extend(new_rows)
                else:
                    warnings.append(
                        f"Page {page_num}: no text extracted (possibly scanned)."
                    )

            # Convert RawRows to BudgetItemCreate objects
            for row in raw_rows:
                amount = parse_greek_amount(row.amount_str)

                # Only warn if the cell looked numeric but failed to parse.
                # Silent failures on purely textual cells (notes, dates) are
                # expected and would produce too many false-positive warnings.
                if amount is None and row.amount_str.strip():
                    if _AMOUNT_RE.search(row.amount_str):
                        warnings.append(
                            f"Page {page_num}: could not parse amount "
                            f"'{row.amount_str[:30]}' for '{row.description[:40]}'"
                        )

                confidence = compute_confidence(
                    amount, row.code, row.description, row.amount_str
                )
                confidences.append(confidence)

                # civic_label() translates the raw Greek category heading into
                # a plain-English label stored in subcategory for the dashboard.
                civic = civic_label(row.category, row.code)

                items.append(BudgetItemCreate(
                    document_id      = document_id,
                    item_type        = classify_item(row.description, doc_type),
                    category         = row.category,
                    subcategory      = civic,
                    code             = row.code,
                    description      = row.description,
                    amount           = amount,
                    source_page      = row.source_page,
                    raw_text         = row.raw_text,
                    confidence_score = confidence,
                ))

    # If we found no text at all, the file is almost certainly scanned.
    # Raising here is better than silently returning zero items.
    if total_text_chars == 0 and not items:
        raise ScannedPDFError(
            f"'{pdf_path.name}' appears to be a scanned (image-only) PDF. "
            "pdfplumber requires a text layer. "
            "Pre-process with: ocrmypdf input.pdf output.pdf"
        )

    with_amount = sum(1 for i in items if i.amount is not None)
    avg_conf    = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return ExtractionResult(
        items                = items,
        page_count           = page_count,
        warnings             = warnings,
        items_with_amount    = with_amount,
        items_without_amount = len(items) - with_amount,
        avg_confidence       = avg_conf,
    )