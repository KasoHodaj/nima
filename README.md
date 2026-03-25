Nima

A Python tool for extracting and visualising budget data from Greek municipal PDFs.
Stack

Extraction — pdfplumber

Storage — SQLite + SQLModel

Search — FTS5 full-text index

API — FastAPI

Dashboard — HTML + Chart.js

Quickstart
bashpip install pdfplumber sqlmodel fastapi uvicorn typer rich

python -m nima init

python -m nima ingest budget.pdf --municipality "Name" --year 2024

python -m nima serve

Open dashboard/index.html in your browser.

Related

OpenCouncil Diavgeia
