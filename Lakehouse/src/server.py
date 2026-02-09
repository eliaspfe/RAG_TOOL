import os
import shutil
from pathlib import Path
from uuid import uuid4
from typing import List, Tuple

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from . import db
from .pipeline import ensure_structure, run_ingest, run_extract, run_chunk


app = FastAPI(title="DuckLake Ingestion Service")


class IngestRequest(BaseModel):
    file_path: str


def _get_db_path() -> Path:
    return Path(os.getenv("DUCKDB_PATH", "/app/DuckLake/ducklake.duckdb"))


def _get_upload_dir() -> Path:
    return Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))


def _get_embedding_service_url() -> str:
    return os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")


def _fetch_embedding_model_info() -> Tuple[str, int]:
    response = requests.get(f"{_get_embedding_service_url()}/model-info", timeout=30)
    response.raise_for_status()
    info = response.json()
    return info["model_name"], int(info["dimension"])


def _get_embeddings(texts: List[str]) -> List[List[float]]:
    response = requests.post(
        f"{_get_embedding_service_url()}/embed",
        json={"texts": texts},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def _embed_doc_chunks(con, doc_id: int) -> dict:
    print(f"[EMBEDDING] Starting _embed_doc_chunks for doc_id={doc_id}")

    model_name, embedding_dim = _fetch_embedding_model_info()
    if embedding_dim != db.EMBEDDING_DIM:
        print(
            f"[EMBEDDING] WARNUNG: Embedding-Dimension {embedding_dim} weicht von DB-"
            f"Dimension {db.EMBEDDING_DIM} ab. Stelle EMBEDDING_DIM passend ein."
        )

    rows = con.execute(
        """
        SELECT chunk_id, chunk_row_id, doc_id, chunk_text, chunk_config_id, run_id
        FROM gold_chunks WHERE doc_id = ? ORDER BY chunk_index
    """,
        [doc_id],
    ).fetchall()

    if not rows:
        print(f"[EMBEDDING] No chunks found for doc_id={doc_id}")
        return {"inserted": 0, "skipped": 0, "total": 0}

    print(f"[EMBEDDING] Found {len(rows)} chunks, creating embeddings...")
    texts = [row[3] for row in rows]
    embeddings = _get_embeddings(texts)
    print(f"[EMBEDDING] Got {len(embeddings)} embeddings back")

    inserted, skipped = 0, 0
    for (
        chunk_id,
        chunk_row_id,
        doc_id_val,
        chunk_text,
        chunk_config_id,
        run_id,
    ), embedding in zip(rows, embeddings):
        try:
            con.execute(
                """
                INSERT INTO gold_embeddings
                (chunk_id, chunk_row_id, doc_id, embedding_text, embedding,
                 embedding_type, chunk_config_id, run_id, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk_row_id,
                    doc_id_val,
                    chunk_text,
                    embedding,
                    "text",
                    chunk_config_id,
                    run_id,
                    model_name,
                ),
            )
            inserted += 1
        except Exception as e:
            print(f"[EMBEDDING] Error inserting chunk {chunk_id}: {e}")
            skipped += 1

    print(f"[EMBEDDING] Inserted {inserted} chunk embeddings, skipped {skipped}")
    return {"inserted": inserted, "skipped": skipped, "total": len(rows)}


@app.on_event("startup")
def startup_event():
    ensure_structure()
    upload_dir = _get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "db_path": str(_get_db_path())}


@app.post("/ingest")
def ingest_all_uploads():
    upload_dir = _get_upload_dir()
    pdf_files = list(upload_dir.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(
            status_code=400, detail="No PDF files found in upload directory"
        )

    results = []
    con = db.get_connection(_get_db_path())
    try:
        for pdf_path in pdf_files:
            doc_id = run_ingest(con, pdf_path)
            run_extract(con, doc_id, pdf_path)
            run_chunk(con, doc_id)
            _embed_doc_chunks(con, doc_id)
            results.append({"doc_id": doc_id, "file_path": str(pdf_path)})
    finally:
        con.close()

    return {"ingested": results}


@app.post("/ingest-upload")
async def ingest_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    upload_dir = _get_upload_dir()
    tmp_path = upload_dir / f"{uuid4().hex}.pdf"

    try:
        with tmp_path.open("wb") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)

        con = db.get_connection(_get_db_path())
        try:
            doc_id = run_ingest(con, tmp_path)
            run_extract(con, doc_id, tmp_path)
            run_chunk(con, doc_id)
            _embed_doc_chunks(con, doc_id)
        finally:
            con.close()

        return {"doc_id": doc_id, "file_path": str(tmp_path)}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
