import argparse
import logging
import os
from pathlib import Path

from . import db
from .bronze import update_bronze_status, upsert_bronze_document
from .gold import chunk_document, export_chunks_parquet, load_chunk_config
from .silver import extract_and_save
from .utils import ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
# DEFAULT_PDF_PATH = BASE_DIR / "data" / "bronze" / "pdfs" / "google-doc-document.pdf"
DEFAULT_PDF_PATH = Path(os.getenv("DEFAULT_PDF_PATH", ""))
CHUNK_CONFIG_PATH = BASE_DIR / "configs" / "chunk_config.json"
EXPORT_DIR = BASE_DIR / "data" / "gold" / "chunks_parquet"


def ensure_structure() -> None:
    ensure_directories(
        [
            BASE_DIR / "DuckLake",
            BASE_DIR / "data" / "bronze" / "pdfs",
            BASE_DIR / "data" / "silver" / "extracted",
            BASE_DIR / "data" / "gold" / "chunks_parquet",
            BASE_DIR / "configs",
            BASE_DIR / "scripts",
            BASE_DIR / "src",
        ]
    )


def run_ingest(con, pdf_path: Path) -> int:
    result = upsert_bronze_document(con, pdf_path)
    return int(result["doc_id"])


def run_extract(con, doc_id: int, pdf_path: Path) -> int:
    page_count = extract_and_save(con, doc_id, pdf_path)
    update_bronze_status(con, doc_id, "EXTRACTED")
    LOGGER.info("Extracted %s pages for doc_id=%s", page_count, doc_id)
    return page_count


def run_chunk(con, doc_id: int) -> None:
    if not CHUNK_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Chunk config missing at {CHUNK_CONFIG_PATH}")
    config = load_chunk_config(CHUNK_CONFIG_PATH)
    result = chunk_document(con, doc_id, config)
    export_chunks_parquet(con, EXPORT_DIR, result["chunk_config_id"], doc_id)
    update_bronze_status(con, doc_id, "CHUNKED")


def main():
    parser = argparse.ArgumentParser(description="DuckLake PDF pipeline")
    sub = parser.add_subparsers(dest="command")
    run_cmd = sub.add_parser("run", help="Run ingest + extract + chunk")
    ingest_cmd = sub.add_parser("ingest", help="Bronze ingest only")
    extract_cmd = sub.add_parser("extract", help="Extract silver text")
    chunk_cmd = sub.add_parser("chunk", help="Chunk and export gold layer")

    def add_pdf_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pdf",
            type=Path,
            default=DEFAULT_PDF_PATH,
            help="Path to input PDF (default: data/bronze/pdfs/google-doc-document.pdf)",
        )

    add_pdf_arg(parser)
    add_pdf_arg(run_cmd)
    add_pdf_arg(ingest_cmd)
    add_pdf_arg(extract_cmd)
    add_pdf_arg(chunk_cmd)

    args = parser.parse_args()
    ensure_structure()
    pdf_path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found at {pdf_path}")

    con = db.get_connection(BASE_DIR / "DuckLake" / "ducklake.duckdb")

    if args.command in (None, "run"):
        try:
            doc_id = run_ingest(con, pdf_path)
            run_extract(con, doc_id, pdf_path)
            run_chunk(con, doc_id)
        except Exception as exc:
            LOGGER.exception("Pipeline failed: %s", exc)
            if "doc_id" in locals():
                update_bronze_status(con, doc_id, "FAILED", str(exc))
            raise
    elif args.command == "ingest":
        doc_id = run_ingest(con, pdf_path)
        LOGGER.info("Ingested doc_id=%s", doc_id)
    elif args.command == "extract":
        doc_id = run_ingest(con, pdf_path)
        run_extract(con, doc_id, pdf_path)
    elif args.command == "chunk":
        doc_id = run_ingest(con, pdf_path)
        run_chunk(con, doc_id)


if __name__ == "__main__":
    main()
