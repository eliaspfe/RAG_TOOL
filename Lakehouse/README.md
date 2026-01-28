DuckLake: PDF ➜ Bronze/Silver/Gold with DuckDB
================================================

Pipeline für die Verarbeitung von `google-doc-document.pdf` in eine DuckDB-Lakehouse-Struktur mit Bronze/Silver/Gold-Layern.

Struktur
--------
- DuckLake/ducklake.duckdb – persistent Storage
- data/bronze/pdfs/google-doc-document.pdf – Input-PDF
- data/silver/extracted/ – optionale Exporte/Debug
- data/gold/chunks_parquet/ – Parquet-Export der Chunks (partitioniert nach chunk_config_id/doc_id)
- configs/chunk_config.json – Chunking-Config
- src/ – Python-Module (bronze, silver, gold, pipeline, utils)
- scripts/run_pipeline.py – CLI-Entrypoint (alias für `python -m src.pipeline`)

Setup
-----
1) Python 3.11+ and a venv:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) System dependencies for hi_res/table extraction:
   - Poppler (`pdfinfo`): e.g. macOS `brew install poppler`, Debian/Ubuntu `apt install poppler-utils`.
   - Optional OCR (for table images): Tesseract installed on the system (`brew install tesseract` or `apt install tesseract-ocr`), then set `ocr_languages` if needed.
3) PDF liegt bereits unter `data/bronze/pdfs/google-doc-document.pdf` (Pfad muss existieren).

Ausführung
----------
- End-to-end: `python -m src.pipeline run`
- Optional anderes PDF: `python -m src.pipeline run --pdf /path/to/your.pdf`
- Einzelstufen: `python -m src.pipeline ingest|extract|chunk`
- Alternativ: `python scripts/run_pipeline.py run`

Docker
------
- Build: `docker build -t ducklake .`
- Run (sample PDF): `docker run --rm -v "$(pwd)/DuckLake":/app/DuckLake -v "$(pwd)/data":/app/data ducklake run`
- Run with custom PDF: `docker run --rm -v "$(pwd)/DuckLake":/app/DuckLake -v "$(pwd)/data":/app/data ducklake run --pdf /app/data/bronze/pdfs/Table_test.pdf` (Option darf jetzt vor oder nach `run` stehen)
  - Mounting `DuckLake/` and `data/` keeps the database and outputs outside the container. Adapt volumes if you want a different input PDF.

Was passiert
------------
- Bronze: PDF wird gehasht (sha256), Metadaten gespeichert (`bronze_documents`), idempotent über (content_hash, source_uri).
- Silver: Extraktion via `unstructured` (bevorzugt `strategy=hi_res`, fallback `fast`, jeweils mit `infer_table_structure=True`). Fließtext wird gereinigt (Zeilenumbrüche, Silbentrennung, Header/Footer-Detektion) und in `silver_extracted_text` gespeichert. Tabellen werden separat als strukturierte JSON-Repräsentationen (inkl. page/table_index, optional caption/html) in `silver_tables` abgelegt, nicht in den Fließtext geflattet.
- Gold: Textseiten werden in Runs (`pages_per_run`) gemergt und gechunked (target_chars/overlap). Tabellen werden als atomare Chunks mit `chunk_type='table'` in `gold_chunks` referenziert (`table_id`), Text-Chunks mit `chunk_type='text'`. Export als partitioniertes Parquet unter `data/gold/chunks_parquet/`.
- Status-Updates in `bronze_documents.status`: `NEW` → `EXTRACTED` → `CHUNKED`, im Fehlerfall `FAILED` + error_message.

Konfiguration
-------------
`configs/chunk_config.json` (Beispielwerte):
- target_chars: 3000
- overlap_chars: 450
- pages_per_run: 3
- split_strategy: paragraph_sentence_fallback

Acceptance
----------
`python -m src.pipeline run` erzeugt (bei vorhandener PDF):
- DuckLake/ducklake.duckdb mit Tabellen bronze/silver/gold
- Bronze-Eintrag (idempotent bei erneutem Lauf)
- Silver-Seiten (clean_text + raw_text) mit konsistenter Seitennummerierung sowie strukturierte Tabellen (`silver_tables`)
- Gold-Chunks mit stabilem chunk_config_id/run_id, chunk_type (text/table), Parquet-Export
