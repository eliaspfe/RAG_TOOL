import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import duckdb
from bs4 import BeautifulSoup

from .utils import (
    chunk_text_with_overlap,
    canonical_json_dumps,
    compute_chunk_config_id,
    generate_run_id,
    read_json,
    stable_chunk_id,
)

LOGGER = logging.getLogger(__name__)


def load_chunk_config(path: Path) -> Dict[str, Any]:
    return read_json(path)


def _persist_chunk_config(
    con: duckdb.DuckDBPyConnection, chunk_config_id: str, chunk_config: Dict[str, Any]
) -> None:
    con.execute(
        """
        INSERT INTO gold_chunk_configs (chunk_config_id, config_json, created_at)
        VALUES (?, ?, now())
        ON CONFLICT (chunk_config_id) DO NOTHING;
        """,
        [chunk_config_id, canonical_json_dumps(chunk_config)],
    )


def _load_silver_pages(con: duckdb.DuckDBPyConnection, doc_id: int) -> List[Dict[str, Any]]:
    rows = con.execute(
        """
        SELECT part_index, clean_text, text_id
        FROM silver_extracted_text
        WHERE doc_id = ?
        ORDER BY part_index ASC;
        """,
        [doc_id],
    ).fetchall()
    pages: List[Dict[str, Any]] = []
    for part_index, clean_text, text_id in rows:
        pages.append({"page_num": int(part_index), "clean_text": clean_text or "", "text_id": text_id})
    return pages


def _load_silver_tables(con: duckdb.DuckDBPyConnection, doc_id: int) -> List[Dict[str, Any]]:
    rows = con.execute(
        """
        SELECT table_id, page, table_index, table_json, caption
        FROM silver_tables
        WHERE doc_id = ?
        ORDER BY page ASC, table_index ASC;
        """,
        [doc_id],
    ).fetchall()
    tables: List[Dict[str, Any]] = []
    for table_id, page, table_index, table_json, caption in rows:
        try:
            data = json.loads(table_json)
        except Exception:
            data = {"raw": table_json}
        tables.append(
            {
                "table_id": int(table_id),
                "page": int(page),
                "table_index": int(table_index),
                "table_json": data,
                "caption": caption,
            }
        )
    return tables


def _table_chunk_text(table: Dict[str, Any]) -> str:
    table_json = table.get("table_json") or {}
    caption = table.get("caption")
    html = table_json.get("text_as_html")
    text_fallback = table_json.get("text") or table_json.get("text_fallback") or ""

    def normalize_value(val: Any) -> str:
        s = "" if val is None else str(val)
        s = s.strip()
        s = s.replace("\u2212", "-").replace("–", "-").replace("—", "-")
        return s

    headers: List[str] = []
    rows: List[List[str]] = []

    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            table_tag = soup.find("table") or soup
            for tr in table_tag.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                values = [normalize_value(c.get_text(separator=" ", strip=True)) for c in cells]
                if tr.find("th") and not headers:
                    headers = values
                    continue
                rows.append(values)
        except Exception:
            headers = []
            rows = []

    if not rows and text_fallback:
        lines = [ln.strip() for ln in text_fallback.splitlines() if ln.strip()]
        if lines:
            headers = re.split(r"\s{2,}|\t", lines[0])
            headers = [normalize_value(h) for h in headers if normalize_value(h)]
            for line in lines[1:]:
                values = [normalize_value(v) for v in re.split(r"\s{2,}|\t", line) if normalize_value(v)]
                if values:
                    rows.append(values)

    payload = {
        "page": table.get("page"),
        "table_index": table.get("table_index"),
        "caption": caption,
        "headers": headers,
        "sections": [],
        "rows": rows,
    }
    return canonical_json_dumps(payload)


def _build_runs(
    pages: Sequence[Dict[str, Any]], pages_per_run: int
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    pages_per_run = max(1, pages_per_run)
    for i in range(0, len(pages), pages_per_run):
        batch = pages[i : i + pages_per_run]
        if not batch:
            continue
        page_start = batch[0]["page_num"]
        page_end = batch[-1]["page_num"]
        run_text = "\n\n".join(p["clean_text"] for p in batch if p["clean_text"].strip())
        runs.append({"page_start": page_start, "page_end": page_end, "text": run_text, "pages": batch})
    return runs


def chunk_document(
    con: duckdb.DuckDBPyConnection,
    doc_id: int,
    chunk_config: Dict[str, Any],
    run_id: str | None = None,
) -> Dict[str, Any]:
    pages = _load_silver_pages(con, doc_id)
    tables = _load_silver_tables(con, doc_id)
    if not pages:
        raise RuntimeError(f"No silver pages found for doc_id={doc_id}")

    run_id = run_id or generate_run_id()
    chunk_config_id = compute_chunk_config_id(chunk_config)
    _persist_chunk_config(con, chunk_config_id, chunk_config)
    chunker_version = "chunker@0.1.0"
    pages_per_run = int(chunk_config.get("pages_per_run", 3))
    runs = _build_runs(pages, pages_per_run)

    insert_sql = """
        INSERT INTO gold_chunks (
            chunk_id, doc_id, page_start, page_end, text_id,
            table_id, chunk_index, chunk_text, char_start, char_end,
            chunk_config_id, run_id, chunker_version, chunk_type, row_start, row_end, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    chunk_counter = 0
    chunk_index = 0
    for run in runs:
        if not run["text"]:
            continue
        flow_text = re.sub(r"\s+", " ", run["text"]).strip()
        if not flow_text:
            continue
        chunks = chunk_text_with_overlap(flow_text, chunk_config)
        for chunk in chunks:
            chunk_id = stable_chunk_id(
                doc_id=doc_id,
                chunk_config_id=chunk_config_id,
                chunk_index=chunk_index,
                page_start=run["page_start"],
                page_end=run["page_end"],
                char_start=chunk["char_start"],
                char_end=chunk["char_end"],
                chunk_text=chunk["chunk_text"],
            )
            con.execute(
                insert_sql,
                [
                    chunk_id,
                    doc_id,
                    run["page_start"],
                    run["page_end"],
                    run["pages"][0].get("text_id"),
                    None,
                    chunk_index,
                    chunk["chunk_text"],
                    chunk["char_start"],
                    chunk["char_end"],
                    chunk_config_id,
                    run_id,
                    chunker_version,
                    "text",
                    None,
                    None,
                    datetime.utcnow(),
                ],
            )
            chunk_counter += 1
            chunk_index += 1

    for table in tables:
        chunk_text = _table_chunk_text(table)
        char_end = len(chunk_text) if chunk_text else 0
        chunk_id = stable_chunk_id(
            doc_id=doc_id,
            chunk_config_id=chunk_config_id,
            chunk_index=chunk_index,
            page_start=table["page"],
            page_end=table["page"],
            char_start=0,
            char_end=char_end,
            chunk_text=chunk_text,
        )
        con.execute(
            insert_sql,
            [
                chunk_id,
                doc_id,
                table["page"],
                table["page"],
                None,
                table["table_id"],
                chunk_index,
                chunk_text,
                0,
                char_end,
                chunk_config_id,
                run_id,
                chunker_version,
                "table",
                None,
                None,
                datetime.utcnow(),
            ],
        )
        chunk_counter += 1
        chunk_index += 1

    LOGGER.info(
        "Chunked doc_id=%s into %s chunks (config_id=%s run_id=%s)",
        doc_id,
        chunk_counter,
        chunk_config_id,
        run_id,
    )
    return {"chunk_count": chunk_counter, "chunk_config_id": chunk_config_id, "run_id": run_id}


def export_chunks_parquet(
    con: duckdb.DuckDBPyConnection, out_dir: Path, chunk_config_id: str, doc_id: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = str(out_dir)
    con.execute(
        f"""
        COPY (
            SELECT * FROM gold_chunks
            WHERE chunk_config_id = ? AND doc_id = ?
        ) TO '{parquet_dir}/' (
            FORMAT 'parquet',
            PARTITION_BY (chunk_config_id, doc_id),
            OVERWRITE_OR_IGNORE TRUE
        );
        """,
        [chunk_config_id, doc_id],
    )
    LOGGER.info("Exported chunks to %s partitioned by chunk_config_id/doc_id", parquet_dir)
