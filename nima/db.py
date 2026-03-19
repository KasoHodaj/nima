"""
nima.db
-------
Database engine creation, session management, and FTS5 index setup.

Why SQLite?
  Zero infrastructure — no server process, no configuration files.
  A single .db file can be copied, backed up, or shared as a snapshot.
  For a production deployment serving many concurrent users, swap
  create_engine() for a PostgreSQL URL; the rest of the codebase is
  unchanged because SQLModel abstracts the dialect.

Why FTS5?
  SQLite's built-in full-text search engine handles Unicode correctly
  (the 'unicode61' tokeniser folds Greek accents and cases), supports
  ranking, and requires zero external dependencies. For a civic tool
  that must run offline in a municipal office, this matters.

Session pattern:
  get_session() is a generator used as a FastAPI dependency. FastAPI
  calls next() to obtain the session, runs the endpoint, then resumes
  the generator (triggering the context manager's __exit__) to close it.
  This guarantees the session is closed even if the endpoint raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine, text


# Default database file path — relative to the working directory.
# Override by passing db_path explicitly or via NIMA_DB_PATH env var.
DEFAULT_DB_PATH = Path("nima.db")


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """
    Return a SQLAlchemy engine for the given SQLite file path.

    check_same_thread=False is required for SQLite when the engine is
    shared across FastAPI's async worker threads. It is safe here because
    SQLModel sessions are short-lived and not shared between threads.
    """
    url = f"sqlite:///{db_path}"
    return create_engine(url, connect_args={"check_same_thread": False})


def init_db(db_path: Path = DEFAULT_DB_PATH):
    """
    Create all SQLModel tables and the FTS5 full-text search index.

    This function is idempotent — safe to call multiple times.
    Tables and triggers are created with IF NOT EXISTS guards.

    FTS5 content table pattern:
      The virtual table mirrors BudgetItem but does not duplicate data.
      Instead, content='budgetitem' tells FTS5 to read the original rows
      for snippet generation. The trigger keeps the index in sync on insert.

    Returns the engine so callers can reuse it without a second call.
    """
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)

    with engine.begin() as conn:
        # Create the FTS5 virtual table linked to the budgetitem table.
        # unicode61 tokeniser: handles Greek characters, accent folding,
        # and case-insensitive matching correctly out of the box.
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS budgetitem_fts
            USING fts5(
                description,
                raw_text,
                content='budgetitem',
                content_rowid='id',
                tokenize='unicode61'
            )
        """))

        # Insert trigger — fires after every INSERT on budgetitem.
        # Without this, the FTS index would fall out of sync with the table.
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS budgetitem_ai
            AFTER INSERT ON budgetitem BEGIN
                INSERT INTO budgetitem_fts(rowid, description, raw_text)
                VALUES (new.id, new.description, new.raw_text);
            END
        """))

    return engine


def get_session(db_path: Path = DEFAULT_DB_PATH) -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in an endpoint:
        @app.get("/example")
        def example(session: Session = Depends(get_session)):
            ...

    The session is automatically closed when the request completes,
    even if an exception is raised inside the endpoint.
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        yield session