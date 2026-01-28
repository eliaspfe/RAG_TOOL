import os
import shutil
from pathlib import Path
from uuid import uuid4

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


@app.on_event("startup")
def startup_event():
    ensure_structure()
    upload_dir = _get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "db_path": str(_get_db_path())}


@app.post("/ingest")
def ingest_by_path(request: IngestRequest):
    pdf_path = Path(request.file_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")

    con = db.get_connection(_get_db_path())
    try:
        doc_id = run_ingest(con, pdf_path)
        run_extract(con, doc_id, pdf_path)
        run_chunk(con, doc_id)
    finally:
        con.close()

    return {"doc_id": doc_id, "file_path": str(pdf_path)}


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
        finally:
            con.close()

        return {"doc_id": doc_id, "file_path": str(tmp_path)}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
