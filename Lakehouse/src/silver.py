import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import duckdb
from unstructured.documents.elements import PageBreak
from unstructured.partition.pdf import partition_pdf

from .utils import clean_text, detect_repeated_lines

LOGGER = logging.getLogger(__name__)

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


def extract_pages_unstructured(pdf_path: Path) -> List[Dict]:
    """Extract pages using unstructured with per-page grouping."""
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            include_page_breaks=True,
        )
        LOGGER.info("partition_pdf strategy=fast")
    except Exception as exc:  # pragma: no cover - depends on env deps
        raise RuntimeError("fast partition failed. Install requirements and retry.") from exc
    pages: Dict[int, List[str]] = {}
    current_page = 1
    for el in elements:
        if isinstance(el, PageBreak):
            current_page += 1
            continue
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) or current_page
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
    return results


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


def extract_and_save(
    con: duckdb.DuckDBPyConnection, doc_id: int, pdf_path: Path
) -> int:
    pages = extract_pages_unstructured(pdf_path)
    try:
        import unstructured

        version = getattr(unstructured, "__version__", "unknown")
    except Exception:
        version = "unstructured@unknown"
    extractor_version = f"unstructured@{version}"
    persist_silver_pages(con, doc_id, pages, extractor_version)
    return len(pages)
