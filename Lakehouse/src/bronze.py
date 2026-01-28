import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import duckdb

from .utils import compute_file_hash_sha256, file_metadata

LOGGER = logging.getLogger(__name__)


def upsert_bronze_document(
    con: duckdb.DuckDBPyConnection, pdf_path: Path
) -> Dict[str, Optional[str]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    meta = file_metadata(pdf_path)
    content_hash = compute_file_hash_sha256(pdf_path)
    source_uri = str(pdf_path)
    LOGGER.info("Ingesting PDF %s (hash=%s)", source_uri, content_hash)

    existing = con.execute(
        """
        SELECT doc_id, status FROM bronze_documents
        WHERE content_hash = ? AND source_uri = ? LIMIT 1;
        """,
        [content_hash, source_uri],
    ).fetchone()

    if existing:
        doc_id, status = existing
        LOGGER.info("Document already ingested with doc_id=%s status=%s", doc_id, status)
        return {"doc_id": doc_id, "status": status}

    now = datetime.utcnow()
    res = con.execute(
        """
        INSERT INTO bronze_documents (
            source_type, source_uri, content_hash,
            file_size_bytes, modified_time, ingested_at, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING doc_id;
        """,
        [
            "pdf",
            source_uri,
            content_hash,
            meta.get("file_size_bytes"),
            meta.get("modified_time"),
            now,
            "NEW",
        ],
    ).fetchone()
    LOGGER.info("Inserted bronze document doc_id=%s", res[0])
    return {"doc_id": res[0], "status": "NEW"}


def update_bronze_status(
    con: duckdb.DuckDBPyConnection, doc_id: int, status: str, error_message: Optional[str] = None
) -> None:
    con.execute(
        "UPDATE bronze_documents SET status = ?, error_message = ? WHERE doc_id = ?;",
        [status, error_message, doc_id],
    )
    LOGGER.info("Updated doc_id=%s to status=%s", doc_id, status)
