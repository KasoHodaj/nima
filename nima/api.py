"""
nima.api
--------
FastAPI application — the HTTP interface between the database and the dashboard.

Endpoint design principles:
  - All endpoints are read-only (GET). Nima ingests data via the CLI,
    not via the API. This simplifies authentication requirements and
    makes the API safe to expose on a local network.
  - Responses use *Read schemas, never raw SQLModel table objects.
    This prevents accidental exposure of internal fields.
  - Pagination is explicit (skip / limit) rather than cursor-based,
    which is simpler for a civic tool where datasets are small.
  - Full-text search uses SQLite FTS5 via a raw SQL query because
    SQLModel does not yet support FTS5 expressions in its ORM layer.

CORS:
  allow_origins=["*"] is intentional for a local-only deployment.
  The dashboard is served from a different port (3000) so CORS headers
  are needed. Tighten this in a production deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, col, func, select, text

from nima.db import DEFAULT_DB_PATH, get_engine, init_db
from nima.models import (
    BudgetItem,
    BudgetItemRead,
    CategorySummary,
    Document,
    DocumentType,
    Municipality,
    MunicipalityRead,
    MunicipalitySummary,
)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Nima API",
    description = "Unthreading public municipal data for civic clarity.",
    version     = "0.1.0",
    docs_url    = "/docs",   # Swagger UI — useful during development
    redoc_url   = "/redoc",  # ReDoc alternative — cleaner for sharing
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Restrict to specific origins in production
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    """
    Resolve the database file path from the environment.

    The CLI sets NIMA_DB_PATH before calling uvicorn so the API uses
    the same database file that was initialised and populated via the CLI.
    Falls back to DEFAULT_DB_PATH if the variable is not set.
    """
    return Path(os.environ.get("NIMA_DB_PATH", str(DEFAULT_DB_PATH)))


def get_session(db_path: Path = Depends(get_db_path)):
    """FastAPI dependency that yields a database session per request."""
    engine = get_engine(db_path)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_municipality_or_404(session: Session, municipality_id: int) -> Municipality:
    """Fetch a municipality by id or raise a 404 HTTPException."""
    muni = session.get(Municipality, municipality_id)
    if not muni:
        raise HTTPException(
            status_code = 404,
            detail      = f"Municipality with id={municipality_id} not found.",
        )
    return muni


def _get_document_or_404(
    session:         Session,
    municipality_id: int,
    year:            int,
    doc_type:        DocumentType,
) -> Document:
    """Fetch a document matching the given filters or raise a 404."""
    doc = session.exec(
        select(Document)
        .where(Document.municipality_id == municipality_id)
        .where(Document.fiscal_year    == year)
        .where(Document.doc_type       == doc_type)
    ).first()
    if not doc:
        raise HTTPException(
            status_code = 404,
            detail      = (
                f"No {doc_type} document found for municipality "
                f"id={municipality_id}, year={year}."
            ),
        )
    return doc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """
    Liveness check.

    Returns the API version so the dashboard can verify it is talking
    to a compatible server before making data requests.
    """
    return {"status": "ok", "version": "0.1.0"}


@app.get("/municipalities", response_model=list[MunicipalityRead], tags=["Municipalities"])
def list_municipalities(session: Session = Depends(get_session)):
    """
    Return all municipalities in the database.

    The dashboard calls this on startup to populate the municipality
    dropdown. If only one municipality exists, the dashboard auto-selects
    it and triggers a data load immediately.
    """
    return session.exec(select(Municipality)).all()


@app.get(
    "/municipalities/{municipality_id}/summary",
    response_model = MunicipalitySummary,
    tags           = ["Data"],
)
def get_summary(
    municipality_id: int,
    year:            int          = Query(..., description="Fiscal year, e.g. 2024"),
    doc_type:        DocumentType = Query(DocumentType.BUDGET),
    session:         Session      = Depends(get_session),
):
    """
    Return a category-level spending breakdown for one municipality and year.

    This is the primary data source for the dashboard bar and doughnut
    charts. Categories are sorted by total_amount descending so the
    largest spending areas appear first.

    The response includes both the raw Greek category names (category)
    and plain-English civic labels (from subcategory) so the dashboard
    can offer a citizen-friendly toggle.
    """
    muni = _get_municipality_or_404(session, municipality_id)
    doc  = _get_document_or_404(session, municipality_id, year, doc_type)

    # Aggregate amounts by category using a single SQL query.
    # We exclude NULL amounts to avoid skewing totals with unextracted rows.
    rows = session.exec(
        select(
            BudgetItem.category,
            func.sum(BudgetItem.amount).label("total_amount"),
            func.count(BudgetItem.id).label("item_count"),
        )
        .where(BudgetItem.document_id == doc.id)
        .where(BudgetItem.amount.isnot(None))
        .group_by(BudgetItem.category)
        .order_by(col("total_amount").desc())
    ).all()

    categories = [
        CategorySummary(
            category     = row.category or "Uncategorised",
            total_amount = row.total_amount or 0.0,
            item_count   = row.item_count,
        )
        for row in rows
    ]

    return MunicipalitySummary(
        municipality = MunicipalityRead.model_validate(muni),
        fiscal_year  = year,
        doc_type     = doc_type,
        total_amount = sum(c.total_amount for c in categories),
        item_count   = sum(c.item_count   for c in categories),
        categories   = categories,
    )


@app.get(
    "/municipalities/{municipality_id}/items",
    response_model = list[BudgetItemRead],
    tags           = ["Data"],
)
def list_items(
    municipality_id: int,
    year:            int                = Query(..., description="Fiscal year"),
    doc_type:        DocumentType       = Query(DocumentType.BUDGET),
    search:          Optional[str]      = Query(None, description="Full-text search query"),
    category:        Optional[str]      = Query(None, description="Filter by exact category name"),
    skip:            int                = Query(0,   ge=0),
    limit:           int                = Query(50,  le=200),
    session:         Session            = Depends(get_session),
):
    """
    Return budget items with optional full-text search and category filter.

    Full-text search uses SQLite FTS5 with the unicode61 tokeniser, which
    handles Greek characters and accent folding correctly. Search results
    are ordered by relevance (FTS5 rank).

    Standard filtered queries (no search term) are ordered by id, which
    corresponds to the order items were extracted from the PDF.
    """
    muni = _get_municipality_or_404(session, municipality_id)
    doc  = _get_document_or_404(session, municipality_id, year, doc_type)

    if search:
        # FTS5 match query — must be raw SQL because SQLModel does not
        # expose FTS5 MATCH expressions in its query builder.
        result = session.exec(
            text("""
                SELECT bi.* FROM budgetitem bi
                JOIN budgetitem_fts fts ON bi.id = fts.rowid
                WHERE fts.budgetitem_fts MATCH :query
                  AND bi.document_id = :doc_id
                ORDER BY rank
                LIMIT :limit OFFSET :skip
            """),
            params={"query": search, "doc_id": doc.id, "limit": limit, "skip": skip},
        ).all()
        return [BudgetItemRead.model_validate(dict(r._mapping)) for r in result]

    # Standard ORM query with optional category filter
    stmt = (
        select(BudgetItem)
        .where(BudgetItem.document_id == doc.id)
    )
    if category:
        stmt = stmt.where(BudgetItem.category == category)

    return session.exec(stmt.offset(skip).limit(limit)).all()


@app.get(
    "/compare",
    response_model = list[MunicipalitySummary],
    tags           = ["Data"],
)
def compare_municipalities(
    ids:      str          = Query(..., description="Comma-separated municipality ids, e.g. 1,2,3"),
    year:     int          = Query(...),
    doc_type: DocumentType = Query(DocumentType.BUDGET),
    session:  Session      = Depends(get_session),
):
    """
    Return summary data for multiple municipalities side by side.

    Designed to power a comparison bar chart in the dashboard. Missing
    municipalities (no document for the requested year) are silently
    skipped so a partial comparison is still useful.

    Maximum 10 municipalities per request to prevent accidental
    denial-of-service from very large comparisons.
    """
    try:
        id_list = [int(i.strip()) for i in ids.split(",") if i.strip()]
    except ValueError:
        raise HTTPException(
            status_code = 400,
            detail      = "ids must be a comma-separated list of integers.",
        )

    if len(id_list) > 10:
        raise HTTPException(
            status_code = 400,
            detail      = "Maximum 10 municipalities per comparison request.",
        )

    results = []
    for mid in id_list:
        try:
            summary = get_summary(
                municipality_id = mid,
                year            = year,
                doc_type        = doc_type,
                session         = session,
            )
            results.append(summary)
        except HTTPException:
            # Skip municipalities with no data for this year — a partial
            # comparison is more useful than an error.
            pass

    return results