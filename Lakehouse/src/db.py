import logging
import os
from pathlib import Path

import duckdb

LOGGER = logging.getLogger(__name__)

# DEFAULT_DB_PATH = Path("DuckLake/ducklake.duckdb")
DEFAULT_DB_PATH = Path(os.getenv("DUCKDB_PATH", "/app/DuckLake/ducklake.duckdb"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads=4")
    run_migrations(con)
    return con


def run_migrations(con: duckdb.DuckDBPyConnection) -> None:
    LOGGER.info("Ensuring DuckLake schema exists")
    try:
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")
        LOGGER.info("DuckDB VSS extension loaded")
    except Exception as exc:
        LOGGER.warning("DuckDB VSS extension could not be loaded: %s", exc)

    con.execute("CREATE SEQUENCE IF NOT EXISTS bronze_documents_seq;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze_documents (
            doc_id INTEGER PRIMARY KEY DEFAULT nextval('bronze_documents_seq'),
            source_type VARCHAR NOT NULL,
            source_uri VARCHAR NOT NULL,
            content_hash VARCHAR NOT NULL,
            file_size_bytes BIGINT,
            modified_time TIMESTAMP,
            ingested_at TIMESTAMP NOT NULL DEFAULT now(),
            status VARCHAR NOT NULL,
            error_message VARCHAR
        );
        """
    )
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bronze_unique ON bronze_documents(content_hash, source_uri);"
    )
    con.execute("CREATE SEQUENCE IF NOT EXISTS silver_extracted_text_seq;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS silver_extracted_text (
            text_id INTEGER PRIMARY KEY DEFAULT nextval('silver_extracted_text_seq'),
            doc_id INTEGER NOT NULL,
            part_type VARCHAR NOT NULL,
            part_index INTEGER NOT NULL,
            raw_text TEXT,
            clean_text TEXT,
            extractor_version VARCHAR NOT NULL,
            extracted_at TIMESTAMP NOT NULL,
            UNIQUE(doc_id, part_index, extractor_version)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_silver_doc_id ON silver_extracted_text(doc_id);")
    con.execute("CREATE SEQUENCE IF NOT EXISTS gold_chunks_seq;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_chunks (
            chunk_row_id INTEGER PRIMARY KEY DEFAULT nextval('gold_chunks_seq'),
            chunk_id VARCHAR NOT NULL,
            doc_id INTEGER NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            text_id INTEGER,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            chunk_config_id VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
            chunker_version VARCHAR NOT NULL,
            chunk_type VARCHAR NOT NULL DEFAULT 'text',
            row_start INTEGER,
            row_end INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE(doc_id, chunk_config_id, run_id, chunk_index)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold_chunk_configs (
            chunk_config_id VARCHAR PRIMARY KEY,
            config_json TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )
    con.execute(
        "ALTER TABLE gold_chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR DEFAULT 'text';"
    )
    con.execute("ALTER TABLE gold_chunks ADD COLUMN IF NOT EXISTS row_start INTEGER;")
    con.execute("ALTER TABLE gold_chunks ADD COLUMN IF NOT EXISTS row_end INTEGER;")

    con.execute("CREATE SEQUENCE IF NOT EXISTS gold_embeddings_seq;")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS gold_embeddings (
            embedding_id INTEGER PRIMARY KEY DEFAULT nextval('gold_embeddings_seq'),
            chunk_row_id INTEGER NOT NULL,
            chunk_id VARCHAR NOT NULL,
            doc_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            model_name VARCHAR NOT NULL,
            embedding FLOAT[{EMBEDDING_DIM}] NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE(chunk_id, model_name)
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_gold_embeddings_chunk_id ON gold_embeddings(chunk_id);")
    try:
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_gold_embeddings_hnsw ON gold_embeddings USING HNSW (embedding) WITH (metric='cosine');"
        )
    except Exception as exc:
        LOGGER.warning("HNSW index for gold_embeddings could not be created: %s", exc)
