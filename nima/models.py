"""
nima.models
-----------
Database schema and API response shapes for Nima.

Architecture decision — three-layer model pattern:
  *Base   — shared fields used by both table and API shapes
  *       — the SQLite table (table=True); never returned directly by the API
  *Create — accepted as input when inserting a new record
  *Read   — returned by API endpoints (always includes the generated id)

This separation means the API never accidentally exposes internal fields
(e.g. raw_text on a summary endpoint) and makes the codebase easier to
extend without breaking existing consumers.

Storage choice:
  SQLite + SQLModel is intentional for zero-infrastructure deployment.
  Any municipality can run Nima on a laptop without a PostgreSQL server.
  A future production deployment would swap the engine URL only.

Traceability:
  Every BudgetItem stores raw_text (the original extracted string) and
  source_page. This lets anyone verify a figure against the original PDF
  without re-running the extractor — a core requirement for civic tools.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ItemType(str, Enum):
    """
    Broad semantic classification of a single budget line.

    Using str as the mixin ensures JSON serialisation works out of the box
    and the value is human-readable in the database (not an integer code).
    """
    REVENUE   = "revenue"    # esodo — money coming in
    EXPENSE   = "expense"    # dapani — money going out (operational)
    TECHNICAL = "technical"  # techniko ergo — capital project or study
    UNKNOWN   = "unknown"    # fallback when heuristics cannot classify


class DocumentType(str, Enum):
    """
    The type of official municipal document a PDF represents.

    Greek municipalities publish two primary document types each year:
    - Budget (proypologismos): all revenues and expenses
    - Technical Programme (techniko programma): infrastructure project list
    """
    BUDGET              = "budget"
    TECHNICAL_PROGRAMME = "technical_programme"


# ---------------------------------------------------------------------------
# Municipality
# ---------------------------------------------------------------------------

class MunicipalityBase(SQLModel):
    """Shared fields for all municipality representations."""

    name: str = Field(
        index=True,
        description="Official Greek name, e.g. Dimos Athinaion",
    )
    name_latin: Optional[str] = Field(
        default=None,
        description="Romanised name used in URL slugs, e.g. athens",
    )
    region: Optional[str] = Field(
        default=None,
        description="Administrative region (Perifereia)",
    )
    population: Optional[int] = Field(
        default=None,
        description="Latest known population from census",
    )


class Municipality(MunicipalityBase, table=True):
    """SQLite table. One row per municipality."""

    id: Optional[int] = Field(default=None, primary_key=True)

    # UTC timestamp — stored as UTC, displayed in local time by the frontend.
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MunicipalityCreate(MunicipalityBase):
    """Input schema when registering a new municipality via the CLI."""
    pass


class MunicipalityRead(MunicipalityBase):
    """API response shape — includes the auto-generated id and timestamp."""
    id: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentBase(SQLModel):
    """
    One Document represents one ingested PDF file.

    Keeping Document separate from BudgetItem means we can re-ingest a
    corrected PDF (producing a new Document) without losing the old one,
    which supports longitudinal comparison across years.
    """

    municipality_id: int = Field(foreign_key="municipality.id")
    doc_type:        DocumentType
    fiscal_year:     int          = Field(description="The year the document covers, e.g. 2024")
    source_filename: str
    page_count:      Optional[int]   = Field(default=None)
    ingested_at:     datetime        = Field(default_factory=datetime.utcnow)
    notes:           Optional[str]   = Field(default=None)


class Document(DocumentBase, table=True):
    """SQLite table. One row per ingested PDF."""
    id: Optional[int] = Field(default=None, primary_key=True)


class DocumentCreate(DocumentBase):
    """Input schema when registering a new document."""
    pass


class DocumentRead(DocumentBase):
    """API response shape."""
    id: int


# ---------------------------------------------------------------------------
# BudgetItem
# ---------------------------------------------------------------------------

class BudgetItemBase(SQLModel):
    """
    One BudgetItem is a single extracted row from a municipal PDF.

    Field design rationale:
    - category:    the raw heading as it appears in the PDF (Greek, messy)
    - subcategory: a normalised plain-English civic label derived from
                   category (see extractor.civic_label). This is what the
                   dashboard displays to citizens.
    - confidence_score: extraction reliability signal (0.0 to 1.0). Surfaced
                   in the API so the frontend can warn users when data is
                   uncertain rather than silently presenting bad numbers.
    - raw_text:    the verbatim extracted string before any parsing. Stored
                   for full traceability so a citizen can verify any figure
                   against the original PDF.
    """

    document_id: int = Field(foreign_key="document.id")

    # Classification
    item_type:   ItemType      = Field(default=ItemType.UNKNOWN)
    category:    Optional[str] = Field(
        default=None,
        index=True,
        description="Raw category heading from the PDF",
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Normalised plain-English civic label",
    )

    # Core content
    code:        Optional[str]   = Field(default=None, description="KAE / KA budget code")
    description: str             = Field(description="Project or line-item description")
    amount:      Optional[float] = Field(default=None, description="Amount in EUR")

    # Provenance — never remove these fields; they are the audit trail
    source_page:      Optional[int] = Field(default=None)
    raw_text:         Optional[str] = Field(default=None)
    confidence_score: float         = Field(
        default=0.5,
        description="Extraction confidence: 0.0 = unreliable, 1.0 = verified",
    )


class BudgetItem(BudgetItemBase, table=True):
    """SQLite table. One row per extracted budget line."""
    id: Optional[int] = Field(default=None, primary_key=True)


class BudgetItemCreate(BudgetItemBase):
    """Input schema used by the extractor when inserting items."""
    pass


class BudgetItemRead(BudgetItemBase):
    """API response shape."""
    id: int


# ---------------------------------------------------------------------------
# Aggregate response schemas
# These are never stored in the database — computed on the fly by the API.
# ---------------------------------------------------------------------------

class CategorySummary(SQLModel):
    """Spending total for one category within a document."""
    category:     str
    total_amount: float
    item_count:   int


class MunicipalitySummary(SQLModel):
    """
    Full summary for one municipality / year / document type.
    Primary payload consumed by the dashboard chart endpoints.
    """
    municipality: MunicipalityRead
    fiscal_year:  int
    doc_type:     DocumentType
    total_amount: float
    item_count:   int
    categories:   list[CategorySummary]