from dotenv import load_dotenv
import duckdb
import os
import uuid
import hashlib
import fitz  # PyMuPDF
import re
from typing import List
from datetime import datetime
import requests
from openai import OpenAI


class DataLake:
    def __init__(
        self,
        db_path: str = "./data/rag_database.duckdb",
        embedding_service_url: str = "http://127.0.0.1:8001",
    ):
        load_dotenv(override=True),
        self.db_path = db_path
        embedding_service_url = embedding_service_url
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = self.conn = duckdb.connect(self.db_path)
        self.initialize_db()
        if os.getenv("EMBEDDING_TYPE") == "api":
            self.client = OpenAI()

    def initialize_db(self):
        self.conn.install_extension("vss")
        self.conn.load_extension("vss")
        self.conn.execute(open("initialize_db.sql").read())

    # Bronze Layer
    def ingest_document_bronze(
        self, file_path: str, doc_name: str, source_type: str = "pdf"
    ) -> int:
        """
        Inserts a document into bronze.documents.

        Returns:
            doc_id (int)
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        # --- Generate BIGINT ID ---
        doc_id = uuid.uuid4().int >> 64

        # --- Read file ---
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)

        # --- Dedup Check ---
        existing = self.conn.execute(
            """
            SELECT doc_id
            FROM bronze.documents
            WHERE content_hash = ?
            """,
            [content_hash],
        ).fetchone()

        if existing:
            print("Document already exists in Bronze.")
            return existing[0]

        # --- Insert ---
        self.conn.execute(
            """
            INSERT INTO bronze.documents (
                doc_id,
                doc_name,
                source_type,
                file_path,
                content_hash,
                file_size_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [doc_id, doc_name, source_type, file_path, content_hash, file_size],
        )

        return doc_id

    # Silver Layer
    def extract_text_to_silver(self, doc_id: int):

        row = self.conn.execute(
            "SELECT file_path FROM bronze.documents WHERE doc_id = ?", [doc_id]
        ).fetchone()

        file_path = row[0]

        pdf = fitz.open(file_path)

        full_text = []
        pages = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()

            pages.append({"page_number": page_number, "text": page_text})

            full_text.append(page_text)

        extracted_text = "\n".join(full_text)
        cleaned_text = self._clean_text(extracted_text)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO silver.document_text (
                doc_id,
                extracted_text,
                cleaned_text,
                extraction_method
            )
            VALUES (?, ?, ?, ?)
            """,
            [doc_id, extracted_text, cleaned_text, "pymupdf"],
        )

        return pages

    def chunk_document_to_silver(
        self, doc_id: int, pages, chunk_size: int = 500, overlap: int = 50
    ):
        chunk_size = 500
        overlap = 50
        chunk_counter = 0

        for page in pages:
            page_number = page["page_number"]
            text = self._clean_text(page["text"])

            chunks = self._chunk_text(text, chunk_size, overlap)

            for chunk in chunks:
                chunk_id = uuid.uuid4().int >> 64

                self.conn.execute(
                    """
                    INSERT INTO silver.document_chunks (
                        chunk_id,
                        doc_id,
                        chunk_index,
                        chunk_text,
                        page_number
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [chunk_id, doc_id, chunk_counter, chunk, page_number],
                )
                chunk_counter += 1
        return pages

    # Gold Layer
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Erstellt Embeddings für eine Liste von Texten über den Embedding-Service
        """
        Args:
            texts: Liste von Texten zum Embedden

        Returns:
            Liste von Embedding-Vektoren
        """
        if os.getenv("EMBEDDING_TYPE") == "local":
            try:
                response = requests.post(
                    "http://embedding-service:8001/embed",
                    json={"texts": texts},
                    timeout=300,  # 5 Minuten Timeout für große Batches
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"]
            except Exception as e:
                print(f"[{datetime.now()}] FEHLER beim Abrufen der Embeddings: {e}")
                raise
        elif os.getenv("EMBEDDING_TYPE") == "api":
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small", input=texts, dimensions=384
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(
                    f"[{datetime.now()}] FEHLER beim Abrufen der Embeddings von OpenAI API: {e}"
                )
                raise
        else:
            raise ValueError("Invalid EMBEDDING_TYPE. Must be 'local' or 'api'.")

    def embed_chunks_to_gold(
        self, doc_id: int, batch_size: int = 32, embedding_model: str = "default"
    ):
        """
        Silver chunks → Embeddings → Gold retrieval_chunks
        Includes document_name + page_number
        """

        # --- Already embedded check ---
        existing = self.conn.execute(
            """
            SELECT 1 FROM gold.retrieval_chunks
            WHERE doc_id = ?
            LIMIT 1
            """,
            [str(doc_id)],
        ).fetchone()

        if existing:
            print(f"[Gold] Document {doc_id} already embedded → skipping")
            return

        # --- Get document name from Bronze ---
        doc_row = self.conn.execute(
            """
            SELECT doc_name
            FROM bronze.documents
            WHERE doc_id = ?
            """,
            [doc_id],
        ).fetchone()

        if not doc_row:
            raise ValueError("Document not found in Bronze")

        document_name = doc_row[0]

        # --- Load chunks WITH page number ---
        rows = self.conn.execute(
            """
            SELECT chunk_id, chunk_text, page_number
            FROM silver.document_chunks
            WHERE doc_id = ?
            ORDER BY chunk_index
            """,
            [doc_id],
        ).fetchall()

        if not rows:
            raise ValueError("No chunks found for document")

        chunk_ids = [str(r[0]) for r in rows]
        texts = [r[1] for r in rows]
        page_numbers = [r[2] for r in rows]

        # --- Batch embedding ---
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunk_ids = chunk_ids[i : i + batch_size]
            batch_pages = page_numbers[i : i + batch_size]

            embeddings = self.get_embeddings(batch_texts)

            # --- Insert batch ---
            for cid, text, page, emb in zip(
                batch_chunk_ids, batch_texts, batch_pages, embeddings
            ):
                self.conn.execute(
                    """
                    INSERT INTO gold.retrieval_chunks (
                        chunk_id,
                        doc_id,
                        document_name,
                        page_number,
                        chunk_text,
                        embedding,
                        embedding_model,
                        token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cid,
                        str(doc_id),
                        document_name,
                        page,
                        text,
                        emb,
                        embedding_model,
                        len(text),
                    ],
                )

        print(f"[Gold] Embedded document {doc_id}")

    def process_document(
        self,
        file_path: str,
        doc_name: str,
        source_type: str = "pdf",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        """
        Full pipeline:
        Bronze → Silver Extraction → Silver Chunking

        Returns:
            doc_id
        """

        # -----------------
        # Bronze
        # -----------------
        doc_id = self.ingest_document_bronze(
            file_path=file_path, doc_name=doc_name, source_type=source_type
        )
        if self._is_document_processed(doc_id):
            print(f"Document {doc_id} already processed → skipping Silver pipeline.")
            return doc_id

        # -----------------
        # Silver Extraction
        # -----------------
        pages = self.extract_text_to_silver(doc_id)

        # -----------------
        # Silver Chunking
        # -----------------
        self.chunk_document_to_silver(
            doc_id=doc_id, pages=pages, chunk_size=chunk_size, overlap=overlap
        )
        # -----------------
        # Gold Embedding
        # -----------------
        self.embed_chunks_to_gold(doc_id)

        return doc_id

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""

        text = text.replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int, overlap: int):
        """Simple sliding window chunking"""

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def _is_document_processed(self, doc_id: int) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM silver.document_chunks
            WHERE doc_id = ?
            LIMIT 1
            """,
            [doc_id],
        ).fetchone()

        return row is not None

    def remove_all_data(self):
        """Löscht alle Daten aus Bronze, Silver und Gold (für Testing)"""
        self.conn.execute("DELETE FROM bronze.documents")
        self.conn.execute("DELETE FROM silver.document_text")
        self.conn.execute("DELETE FROM silver.document_chunks")
        self.conn.execute("DELETE FROM gold.retrieval_chunks")
        print("Alle Daten gelöscht.")
