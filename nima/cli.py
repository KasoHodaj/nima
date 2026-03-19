"""
nima.cli
--------
Command-line interface for Nima — the primary entry point for data ingestion.

Commands
~~~~~~~~
    nima init
        Initialise the SQLite database and FTS5 index.
        Safe to run multiple times (idempotent).

    nima ingest <pdf> --municipality <name> --year <year> [--type <type>]
        Ingest a municipal PDF and store all extracted budget items.
        Creates the municipality record if it does not already exist.

    nima list-municipalities
        Print a table of all municipalities currently in the database.

    nima serve [--host] [--port] [--db]
        Start the FastAPI development server.

Design notes:
  Typer is used instead of argparse because it generates help text
  automatically from type hints and produces a better UX with Rich.
  The CLI is the only place that writes to the database — the API is
  read-only. This separation makes it safe to run the API in a shared
  environment without risk of accidental data modification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import Session, select

from nima.db import DEFAULT_DB_PATH, get_engine, init_db
from nima.extractor import ExtractionError, ScannedPDFError, extract
from nima.models import (
    BudgetItem,
    Document,
    DocumentCreate,
    DocumentType,
    Municipality,
    MunicipalityCreate,
)

# ---------------------------------------------------------------------------
# App and console setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name           = "nima",
    help           = "Nima — Unthreading public data for civic clarity.",
    add_completion = False,   # Disable shell completion for portability
)

# Rich console for coloured terminal output.
# All user-facing messages go through this rather than print() so that
# output formatting is consistent and can be redirected cleanly.
console = Console()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create_municipality(session: Session, name: str) -> Municipality:
    """
    Return the Municipality with the given name, creating it if necessary.

    Using name as the lookup key means re-ingesting a second PDF for the
    same municipality links both documents to the same record without
    requiring the user to remember an integer id.
    """
    existing = session.exec(
        select(Municipality).where(Municipality.name == name)
    ).first()

    if existing:
        return existing

    # Municipality not found — create it with the name provided.
    # Region and population can be updated later via a future edit command.
    new_muni = Municipality.model_validate(MunicipalityCreate(name=name))
    session.add(new_muni)
    session.commit()
    session.refresh(new_muni)
    console.print(f"[green]✓[/green] Created municipality: [bold]{name}[/bold]")
    return new_muni


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def init(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite database file."),
):
    """
    Initialise the database.

    Creates all tables and the FTS5 full-text search index.
    Safe to run multiple times — existing data is not affected.
    """
    init_db(db)
    console.print(f"[green]✓[/green] Database initialised at [bold]{db}[/bold]")


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="Path to the municipal PDF file to ingest."),
    municipality: str = typer.Option(
        ..., "--municipality", "-m",
        help="Municipality name (e.g. 'Δήμος Αθηναίων'). Created if it does not exist.",
    ),
    year: int = typer.Option(
        ..., "--year", "-y",
        help="Fiscal year the document covers (e.g. 2024).",
    ),
    doc_type: DocumentType = typer.Option(
        DocumentType.BUDGET, "--type", "-t",
        help="Document type: 'budget' or 'technical_programme'.",
    ),
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite database file."),
):
    """
    Ingest a municipal PDF and store all extracted budget items.

    The PDF must have a text layer (not scanned). If extraction fails
    due to a scanned document, the command prints a helpful message and
    exits without modifying the database.
    """
    if not pdf.exists():
        console.print(f"[red]✗[/red] File not found: {pdf}")
        raise typer.Exit(code=1)

    engine = get_engine(db)

    with Session(engine) as session:
        muni = _get_or_create_municipality(session, municipality)

        # Insert the Document record first so we have an id for the extractor.
        # If extraction fails we roll back this record before exiting.
        doc = Document.model_validate(DocumentCreate(
            municipality_id = muni.id,
            doc_type        = doc_type,
            fiscal_year     = year,
            source_filename = pdf.name,
        ))
        session.add(doc)
        session.commit()
        session.refresh(doc)

        console.print(f"[cyan]→[/cyan] Extracting from [bold]{pdf.name}[/bold] …")

        try:
            result = extract(pdf, document_id=doc.id, doc_type=doc_type)

        except ScannedPDFError as exc:
            console.print(f"[red]✗ Scanned PDF:[/red] {exc}")
            # Roll back the document record — partial data is worse than none
            session.delete(doc)
            session.commit()
            raise typer.Exit(code=1)

        except ExtractionError as exc:
            console.print(f"[red]✗ Extraction error:[/red] {exc}")
            session.delete(doc)
            session.commit()
            raise typer.Exit(code=1)

        # Persist all extracted items in a single transaction
        for item_create in result.items:
            session.add(BudgetItem.model_validate(item_create))

        # Update the page count now that extraction is complete
        doc.page_count = result.page_count
        session.add(doc)
        session.commit()

    # Report extraction quality metrics to the user
    console.print(
        f"[green]✓[/green] Ingested [bold]{len(result.items)}[/bold] items "
        f"from {result.page_count} pages. "
        f"Avg confidence: [bold]{result.avg_confidence:.2f}[/bold]"
    )
    console.print(
        f"   With amount: {result.items_with_amount} | "
        f"Without: {result.items_without_amount}"
    )

    # Show warnings, capped at 10 to avoid flooding the terminal
    if result.warnings:
        console.print(f"[yellow]⚠[/yellow]  {len(result.warnings)} warning(s):")
        for warning in result.warnings[:10]:
            console.print(f"   [dim]{warning}[/dim]")
        if len(result.warnings) > 10:
            console.print(
                f"   [dim]… and {len(result.warnings) - 10} more.[/dim]"
            )


@app.command(name="list-municipalities")
def list_municipalities(
    db: Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite database file."),
):
    """List all municipalities currently in the database."""
    engine = get_engine(db)
    with Session(engine) as session:
        munis = session.exec(select(Municipality)).all()

    if not munis:
        console.print("[dim]No municipalities found. Run 'nima ingest' first.[/dim]")
        return

    table = Table(title="Municipalities", show_lines=True)
    table.add_column("ID",     style="dim", width=5)
    table.add_column("Name",   style="bold")
    table.add_column("Region", style="")

    for muni in munis:
        table.add_row(str(muni.id), muni.name, muni.region or "—")

    console.print(table)


@app.command()
def serve(
    host: str  = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int  = typer.Option(8000,        help="Port to listen on."),
    db:   Path = typer.Option(DEFAULT_DB_PATH, help="Path to the SQLite database file."),
):
    """
    Start the Nima API development server.

    The server runs with --reload enabled, so changes to the source code
    are picked up automatically without restarting. Do not use --reload
    in a production deployment.
    """
    # Pass the database path to the API via an environment variable.
    # This is simpler than a config file for a single-process deployment.
    os.environ["NIMA_DB_PATH"] = str(db)

    import uvicorn

    console.print(
        f"[green]✓[/green] Nima API starting at "
        f"[bold]http://{host}:{port}[/bold] — "
        f"interactive docs at [bold]/docs[/bold]"
    )
    uvicorn.run("nima.api:app", host=host, port=port, reload=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()