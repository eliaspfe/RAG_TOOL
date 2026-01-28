import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import duckdb
from unstructured.documents.elements import PageBreak, Table
from unstructured.partition.pdf import partition_pdf

from .utils import canonical_json_dumps, clean_text, detect_repeated_lines

LOGGER = logging.getLogger(__name__)


def _table_to_record(table: Table, page_num: int, table_index: int) -> Dict[str, Any]:
    meta = getattr(table, "metadata", None)
    text_as_html = getattr(meta, "text_as_html", None)
    caption = getattr(meta, "caption", None)
    meta_dict = {}
    if meta and hasattr(meta, "to_dict"):
        try:
            meta_dict = meta.to_dict()
        except Exception:
            meta_dict = {}
    table_dict: Dict[str, Any] = {"type": "table", "page": page_num, "table_index": table_index}
    if hasattr(table, "to_dict"):
        try:
            table_dict.update(table.to_dict())
        except Exception:
            table_dict["text"] = str(table)
    table_dict["text_as_html"] = text_as_html
    table_dict["caption"] = caption
    table_dict["metadata"] = meta_dict
    table_dict["text_fallback"] = str(table)
    return {
        "page_num": page_num,
        "table_index": table_index,
        "table_json": table_dict,
        "caption": caption,
    }


def _should_skip_element(el) -> bool:
    """Filter headers/footers/footnotes, figure bodies (keep captions), and rotated/marginal items."""
    skip_categories = {
        "PageHeader",
        "Header",
        "PageFooter",
        "Footer",
        "Footnote",
    }
    figure_body_categories = {
        "Image",
        "Figure",
        "Chart",
        "Picture",
    }

    meta = getattr(el, "metadata", None)
    category = getattr(el, "category", None) or getattr(meta, "category", None)

    # Drop known non-content and figure bodies; keep captions/narrative text.
    if category in skip_categories:
        return True
    if category in figure_body_categories:
        return True
    if category == "UncategorizedText" and getattr(meta, "parent_id", None):
        parent = str(getattr(meta, "parent_id"))
        if "figure" in parent.lower() or "image" in parent.lower():
            return True

    if not meta:
        return False

    rotation = getattr(meta, "rotation", None)
    if rotation is None:
        rotation = getattr(meta, "orientation", None)
    try:
        if rotation is not None and abs(float(rotation)) > 1.0:
            return True
    except Exception:
        pass

    # Marginal content: if we have coordinates with layout size, drop text far in the left/right margin.
    try:
        coords = getattr(meta, "coordinates", None)
        if coords and hasattr(coords, "to_dict"):
            coords = coords.to_dict()
        if isinstance(coords, dict):
            points = coords.get("points") or []
            layout_w = coords.get("layout_width") or coords.get("width")
            if points and layout_w:
                xs = [pt[0] for pt in points if isinstance(pt, (list, tuple)) and len(pt) == 2]
                if xs:
                    cx = (min(xs) + max(xs)) / 2.0
                    if layout_w > 0:
                        rel_cx = cx / float(layout_w)
                        if rel_cx < 0.1 or rel_cx > 0.9:
                            return True
    except Exception:
        pass

    return False


def extract_pages_unstructured(pdf_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract pages using unstructured with per-page grouping; keep tables separate."""
    # Enforce hi_res for table-aware extraction; fail loudly if deps are missing.
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            include_page_breaks=True,
            infer_table_structure=True,
            extract_image_block_types=["Table"],
        )
        LOGGER.info("partition_pdf strategy=hi_res (table-aware)")
    except Exception as exc:  # pragma: no cover - depends on env deps
        raise RuntimeError(
            "hi_res partition failed (table detection requires unstructured-inference/layoutparser/torch and optional OCR). "
            "Install requirements and retry."
        ) from exc
    pages: Dict[int, List[str]] = {}
    tables: List[Dict[str, Any]] = []
    current_page = 1
    table_counter: Dict[int, int] = {}
    for el in elements:
        if isinstance(el, PageBreak):
            current_page += 1
            continue
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) or current_page
        if isinstance(el, Table):
            table_idx = table_counter.get(page_num, 0)
            table_counter[page_num] = table_idx + 1
            tables.append(_table_to_record(el, page_num, table_idx))
            current_page = page_num
            continue
        if _should_skip_element(el):
            current_page = page_num
            continue
        pages.setdefault(page_num, []).append(str(el))
        current_page = page_num
    results: List[Dict] = []
    for page_num in sorted(pages.keys()):
        raw = "\n\n".join(pages[page_num]).strip()
        results.append({"page_num": page_num, "raw_text": raw})
    if not results:
        results.append({"page_num": 1, "raw_text": ""})
    LOGGER.info("Extracted %s pages from %s", len(results), pdf_path)
    LOGGER.info("Detected %s tables from %s", len(tables), pdf_path)
    return results, tables


def persist_silver_pages(
    con: duckdb.DuckDBPyConnection,
    doc_id: int,
    pages: Sequence[Dict],
    extractor_version: str,
) -> None:
    raw_texts = [p["raw_text"] for p in pages]
    repeated = detect_repeated_lines(raw_texts)
    now = datetime.utcnow()
    for page in pages:
        clean = clean_text(page["raw_text"], repeated_lines=repeated)
        con.execute(
            """
            INSERT INTO silver_extracted_text (
                doc_id, part_type, part_index, raw_text, clean_text, extractor_version, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, part_index, extractor_version) DO UPDATE SET
                raw_text=excluded.raw_text,
                clean_text=excluded.clean_text,
                extracted_at=excluded.extracted_at,
                part_type=excluded.part_type;
            """,
            [
                doc_id,
                "page",
                int(page["page_num"]),
                page["raw_text"],
                clean,
                extractor_version,
                now,
            ],
        )
    LOGGER.info("Persisted %s silver pages for doc_id=%s", len(pages), doc_id)


def persist_silver_tables(
    con: duckdb.DuckDBPyConnection,
    doc_id: int,
    tables: Sequence[Dict[str, Any]],
) -> None:
    now = datetime.utcnow()
    for table in tables:
        con.execute(
            """
            INSERT INTO silver_tables (
                doc_id, page, table_index, table_json, caption, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, page, table_index) DO UPDATE SET
                table_json=excluded.table_json,
                caption=excluded.caption,
                extracted_at=excluded.extracted_at;
            """,
            [
                doc_id,
                int(table["page_num"]),
                int(table["table_index"]),
                canonical_json_dumps(table["table_json"]),
                table.get("caption"),
                now,
            ],
        )
    LOGGER.info("Persisted %s silver tables for doc_id=%s", len(tables), doc_id)


def extract_and_save(
    con: duckdb.DuckDBPyConnection, doc_id: int, pdf_path: Path
) -> int:
    pages, tables = extract_pages_unstructured(pdf_path)
    try:
        import unstructured

        version = getattr(unstructured, "__version__", "unknown")
    except Exception:
        version = "unstructured@unknown"
    extractor_version = f"unstructured@{version}"
    persist_silver_pages(con, doc_id, pages, extractor_version)
    persist_silver_tables(con, doc_id, tables)
    return len(pages)
