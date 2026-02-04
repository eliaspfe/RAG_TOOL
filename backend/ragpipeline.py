import duckdb
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from datetime import datetime
import os
import uuid


class RagPipeline:
    def __init__(
        self,
        db_path: str = "./data/rag_database.duckdb",
        schema_name: str = "embeddings",
    ):
        load_dotenv(override=True)
        self.db_path = db_path
        self.schema_name = schema_name
        self.client = OpenAI()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self.initialize_db()

    def initialize_db(self):
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        # Schema anlegen
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")

        # Quell-Tabelle (falls nicht existiert)
        self.conn.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.gold_chunks (
            chunk_id VARCHAR PRIMARY KEY,
            chunk_text VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        )

        # Embedding-Tabelle
        self.conn.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.gold_chunk_embeddings (
            chunk_id VARCHAR PRIMARY KEY,
            chunk_text VARCHAR,
            embedding FLOAT[1536],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        )
        self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
        self.conn.execute(
            f"""
        CREATE INDEX IF NOT EXISTS gold_chunk_embeddings_hnsw
        ON {self.schema_name}.gold_chunk_embeddings
        USING HNSW (embedding);
        """
        )

    def load_and_embed_chunks(self):
        # Nur Chunks ohne Embedding laden
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, chunk_text
            FROM {self.schema_name}.gold_chunks
            WHERE chunk_id NOT IN (
                SELECT chunk_id FROM {self.schema_name}.gold_chunk_embeddings
            )
        """
        ).fetchall()

        print(f"{len(rows)} Chunks werden embedded...")

        for chunk_id, text in rows:
            response = self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            )

            embedding = response.data[0].embedding

            self.conn.execute(
                f"""
                INSERT INTO {self.schema_name}.gold_chunk_embeddings
                (chunk_id, chunk_text, embedding)
                VALUES (?, ?, ?)
            """,
                (chunk_id, text, embedding),
            )

        print("Embeddings gespeichert.")

    def pdf_to_chunks(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 100):
        """
        Liest ein PDF ein und teilt den Text in Chunks auf.

        :param pdf_path: Pfad zur PDF-Datei
        :param chunk_size: Länge eines Chunks (in Zeichen)
        :param overlap: Überlappung zwischen Chunks (in Zeichen)
        :return: Liste von Text-Chunks
        """
        reader = PdfReader(pdf_path)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        chunks = []
        start = 0
        text_length = len(full_text)

        while start < text_length:
            end = start + chunk_size
            chunk = full_text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def chunks_to_duckdb(self, chunks):
        """
        Speichert gegebene Chunks in der gold_chunks Tabelle der DuckDB

        Args:
            chunks: Liste von Text-Chunks
        """

        # Tabelle sicherstellen
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.gold_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                chunk_text VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        inserted = 0

        for chunk_text in chunks:

            chunk_id = str(uuid.uuid4())  # VARCHAR laut Schema

            try:
                self.conn.execute(
                    f"""
                    INSERT INTO {self.schema_name}.gold_chunks 
                    (chunk_id, chunk_text)
                    VALUES (?, ?)
                    """,
                    (chunk_id, chunk_text),
                )
                inserted += 1

            except Exception as e:
                print(
                    f"[{datetime.now()}] WARNUNG: Chunk {chunk_id} konnte nicht eingefügt werden: {e}"
                )

        print(f"{inserted} Chunks eingefügt.")

    def pdf_chunk_and_store(self, pdf_path: str):
        """
        Liest ein PDF, erstellt Chunks und speichert sie in DuckDB

        Args:
            pdf_path: Pfad zur PDF-Datei
        """
        print(f"[{datetime.now()}] Lese PDF: {pdf_path}")
        chunks = self.pdf_to_chunks(pdf_path)
        print(f"[{datetime.now()}] Erstelle {len(chunks)} Chunks aus PDF")
        self.chunks_to_duckdb(chunks)
        print(f"[{datetime.now()}] Chunks in DuckDB gespeichert")

    def similarity_search(self, query: str, top_k: int = 5):
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=query
        )

        query_embedding = response.data[0].embedding

        results = self.conn.execute(
            f"""
            SELECT
                chunk_text
            FROM {self.schema_name}.gold_chunk_embeddings
            ORDER BY array_cosine_similarity(
                embedding,
                CAST(? AS FLOAT[1536])
            ) DESC
            LIMIT ?
            """,
            (query_embedding, top_k),
        ).fetchall()

        return [row[0] for row in results]

    def build_prompt(self, query: str, top_k: int = 5):
        """
        Erstellt einen Prompt mit Kontext aus den ähnlichsten Chunks.

        Args:
            query: Nutzerfrage
            top_k: Anzahl ähnlicher Chunks

        Returns:
            Prompt-String
        """

        # Ähnlichste Texte holen
        similar_texts = self.similarity_search(query, top_k=top_k)

        context = "\n\n".join(
            [f"Kontext {i+1}:\n{text}" for i, text in enumerate(similar_texts)]
        )

        prompt = f"""Beantworte jede Frage des Nutzers mit Hilfe des hier übegebenen Kontextes. Sollte die Antwort nicht zu finden sein weise den Nutzer darauf hin, dass die Informationen nicht gefunden werden konnten.
    Gesamter Kontext:
    {context}

    Nutzer Frage:
    {query}
    """

        return prompt
