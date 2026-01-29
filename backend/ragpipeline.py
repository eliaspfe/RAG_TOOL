import os
import duckdb
from typing import List, Tuple, Optional
from datetime import datetime
import json
import requests
from PyPDF2 import PdfReader


class RagPipeline:

    def __init__(
        self,
        db_path: str = "./data/rag_database.duckdb",
        embedding_service_url: str = "http://embedding-service:8001",
        schema_name: str = "embeddings",
    ):
        """
        Initialisiert die RAG Pipeline

        Args:
            db_path: Pfad zur DuckDB Datenbank (gleiche DB wie DuckLake, Embeddings in separatem Schema)
            embedding_service_url: URL des Embedding-Service-Containers
            schema_name: Name des DuckDB Schemas für Embeddings (separate von gold/silver/bronze)
        """
        self.db_path = db_path
        self.embedding_service_url = embedding_service_url
        self.schema_name = schema_name
        self.read_only_db = os.getenv("DUCKDB_READ_ONLY", "false").lower() == "true"
        self.embedding_dim = None
        self.model_name = None
        self.llm_model_name = None
        self.conn = None

        # Initialisiere bei Instanziierung
        self._fetch_model_info()
        self._initialize_db()

    def _fetch_model_info(self):
        """Holt Modell-Informationen vom Embedding-Service und setzt embedding_dim"""
        try:
            response = requests.get(
                f"{self.embedding_service_url}/model-info", timeout=30
            )
            response.raise_for_status()
            info = response.json()

            self.embedding_dim = info["dimension"]
            self.model_name = info["model_name"]

            print(f"[{datetime.now()}] Embedding-Service verbunden:")
            print(f"  - Modell: {self.model_name}")
            print(f"  - Dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER: Kann Modell-Info nicht abrufen: {e}")
            print(f"[{datetime.now()}] Verwende Standard-Dimension: 768")
            self.embedding_dim = 768
            self.model_name = "unknown"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Erstellt Embeddings für eine Liste von Texten über den Embedding-Service
        """
        Args:
            texts: Liste von Texten zum Embedden

        Returns:
            Liste von Embedding-Vektoren
        """
        try:
            response = requests.post(
                f"{self.embedding_service_url}/embed",
                json={"texts": texts},
                timeout=300,  # 5 Minuten Timeout für große Batches
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"]
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER beim Abrufen der Embeddings: {e}")
            raise

    def _initialize_db(self):
        """Initialisiert die DuckDB Verbindung und erstellt Embedding-Tabellen"""
        print(f"[{datetime.now()}] Verbinde mit DuckDB: {self.db_path}")

        # Erstelle Verzeichnis falls nicht vorhanden
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Optional read-only Modus, um Schreib-Locks zu vermeiden
        self.conn = duckdb.connect(self.db_path, read_only=self.read_only_db)

        if self.read_only_db:
            print(
                f"[{datetime.now()}] DuckDB im Read-Only Modus geöffnet (keine Schema-Änderungen)"
            )
            return

        # Installiere und lade VSS Extension für Vektor-Ähnlichkeitssuche
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")

        # Erstelle Schema für Embeddings
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
        print(f"[{datetime.now()}] Schema '{self.schema_name}' erstellt/gefunden")

        # Erstelle eine Tabelle für Gold Chunks, falls nicht vorhanden
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.gold_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                chunk_row_id INTEGER,
                chunk_text VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        # Prüfe ob Tabellen bereits existieren und ob Dimension passt
        existing_tables = self.conn.execute(
            f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{self.schema_name}'
        """
        ).fetchall()

        if existing_tables:
            print(
                f"[{datetime.now()}] Bestehende Tabellen gefunden in Schema '{self.schema_name}'"
            )
            # TODO: Hier könnte man die Dimension validieren

        # Erstelle Sequence im Schema
        self.conn.execute(
            f"""
            CREATE SEQUENCE IF NOT EXISTS {self.schema_name}.chunk_embeddings_seq START 1;
        """
        )

        # Erstelle Tabelle für Chunk-Embeddings (verknüpft mit gold_chunks)
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.chunk_embeddings (
                embedding_id INTEGER PRIMARY KEY DEFAULT nextval('{self.schema_name}.chunk_embeddings_seq'),
                chunk_id VARCHAR NOT NULL,
                chunk_row_id INTEGER NOT NULL,
                doc_id INTEGER NOT NULL,
                embedding_text VARCHAR NOT NULL,
                embedding FLOAT[{self.embedding_dim}] NOT NULL,
                embedding_type VARCHAR DEFAULT 'text',
                chunk_config_id VARCHAR,
                run_id VARCHAR,
                model_name VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chunk_id, embedding_type, model_name)
            )
        """
        )

        # Erstelle Tabelle für Tabellen-Embeddings (verknüpft mit silver_tables)
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.table_embeddings (
                embedding_id INTEGER PRIMARY KEY,
                table_id INTEGER NOT NULL,
                doc_id INTEGER NOT NULL,
                page INTEGER NOT NULL,
                table_index INTEGER NOT NULL,
                row_index INTEGER,
                section VARCHAR,
                embedding_text VARCHAR NOT NULL,
                embedding FLOAT[{self.embedding_dim}] NOT NULL,
                model_name VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(table_id, row_index, model_name)
            )
        """
        )

        print(
            f"[{datetime.now()}] Datenbank und Schema '{self.schema_name}' initialisiert!"
        )
        print(f"[{datetime.now()}] Embedding-Dimension: {self.embedding_dim}")

    # ============================================================================
    # NOAH'S BEREICH: PDF-Extraktion und DuckLake Pipeline
    # ============================================================================

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

    def chunks_to_duckdb(
        self,
        chunks: List[str],
    ):
        """
        Speichert gegebene Chunks in der gold_chunks Tabelle der DuckDB

        Args:
            chunks: Liste von Text-Chunks
            doc_id: ID des zugehörigen Dokuments

        """
        inserted = 0
        for idx, chunk_text in enumerate(chunks):
            chunk_id = idx + 1  # Einfache Chunk-ID (kann angepasst werden)
            try:
                self.conn.execute(
                    f"""
                    INSERT INTO {self.schema_name}.gold_chunks 
                    (chunk_id, chunk_row_id, chunk_text)
                    VALUES (?, ?, ?)
                """,
                    (chunk_id, idx, chunk_text),
                )
                inserted += 1
            except Exception as e:
                print(
                    f"[{datetime.now()}] WARNUNG: Chunk {chunk_id} konnte nicht eingefügt werden: {e}"
                )

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

    # ============================================================================
    # FELIX'S BEREICH: Embedding-Funktionalität
    # ============================================================================
    def load_chunks_from_gold(
        self, chunk_config_id: str = None, doc_id: int = None
    ) -> List[dict]:
        """
        Lädt Chunks aus der gold_chunks Tabelle

        Args:
            chunk_config_id: Optional - filtert nach spezifischer Chunking-Konfiguration
            doc_id: Optional - filtert nach spezifischem Dokument

        Returns:
            Liste von Chunk-Dictionaries
        """
        query = "SELECT * FROM gold_chunks WHERE 1=1"
        params = []

        if chunk_config_id:
            query += " AND chunk_config_id = ?"
            params.append(chunk_config_id)

        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)

        query += " ORDER BY doc_id, chunk_index"

        result = self.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]

        chunks = []
        for row in result:
            chunk_dict = dict(zip(columns, row))
            chunks.append(chunk_dict)

        print(f"[{datetime.now()}] {len(chunks)} Chunks aus gold_chunks geladen")
        return chunks

    def load_tables_from_silver(self, doc_id: int = None) -> List[dict]:
        """
        Lädt Tabellen aus der silver_tables Tabelle

        Args:
            doc_id: Optional - filtert nach spezifischem Dokument

        Returns:
            Liste von Tabellen-Dictionaries
        """
        query = "SELECT * FROM silver_tables WHERE 1=1"
        params = []

        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)

        query += " ORDER BY doc_id, page, table_index"

        result = self.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]

        tables = []
        for row in result:
            table_dict = dict(zip(columns, row))
            # table_json liegt bereits als JSON vor
            if table_dict.get("table_json"):
                # Falls es ein String ist, parse es, sonst nutze es direkt
                if isinstance(table_dict["table_json"], str):
                    table_dict["table_data"] = json.loads(table_dict["table_json"])
                else:
                    table_dict["table_data"] = table_dict["table_json"]
            tables.append(table_dict)

        print(f"[{datetime.now()}] {len(tables)} Tabellen aus silver_tables geladen")
        return tables

    def embed_chunks_and_save_to_duckdb(
        self, chunk_config_id: str = None, doc_id: int = None
    ) -> dict:
        """
        Lädt Chunks aus gold_chunks, erstellt Embeddings und speichert sie

        Args:
            chunk_config_id: Optional - filtert nach spezifischer Chunking-Konfiguration
            doc_id: Optional - filtert nach spezifischem Dokument

        Returns:
            Dictionary mit Statistiken (inserted, skipped, total)
        """
        # Lade Chunks aus DuckLake
        chunks = self.load_chunks_from_gold(chunk_config_id, doc_id)

        if not chunks:
            print(f"[{datetime.now()}] WARNUNG: Keine Chunks zum Verarbeiten")
            return {"inserted": 0, "skipped": 0, "total": 0}

        try:
            # Extrahiere Texte für Embeddings
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            # Erstelle Embeddings über Service
            print(
                f"[{datetime.now()}] Erstelle Embeddings für {len(chunk_texts)} Chunks..."
            )
            embeddings = self.get_embeddings(chunk_texts)
            print(f"[{datetime.now()}] Embeddings erstellt: {len(embeddings)}")

            # Speichere in Datenbank
            print(f"[{datetime.now()}] Speichere in DB...")
            inserted = 0
            skipped = 0

            for chunk, embedding in zip(chunks, embeddings):
                try:
                    self.conn.execute(
                        f"""
                        INSERT INTO {self.schema_name}.chunk_embeddings 
                        (chunk_id, chunk_row_id, doc_id, embedding_text, embedding, 
                         embedding_type, chunk_config_id, run_id, model_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            chunk["chunk_id"],
                            chunk["chunk_row_id"],
                            chunk["doc_id"],
                            chunk["chunk_text"],
                            embedding,  # embedding ist bereits eine Liste
                            "text",
                            chunk["chunk_config_id"],
                            chunk["run_id"],
                            self.model_name,
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    # Duplikat oder anderer Fehler
                    skipped += 1

            print(
                f"[{datetime.now()}] In DB gespeichert: {inserted} neu, {skipped} übersprungen"
            )

            return {"inserted": inserted, "skipped": skipped, "total": len(chunks)}

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER beim Embedding: {e}")
            raise

    def format_table_for_embedding(self, table_data: dict) -> List[str]:
        """
        Formatiert eine Tabelle im JSON-Format für Embeddings
        Erstellt separate Embeddings pro Tabellenzeile

        Args:
            table_data: Dictionary mit caption, headers, rows, sections, page, table_index

        Returns:
            Liste von formatierten Strings für Embeddings
        """
        embeddings_texts = []

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        caption = table_data.get("caption")
        page = table_data.get("page")
        table_index = table_data.get("table_index", 0)

        # Header-String für alle Zeilen
        header_str = " | ".join(headers) if headers else ""

        current_section = None

        for row_idx, row in enumerate(rows):
            # Prüfe ob Zeile eine Section ist (nur ein Element und keine Daten)
            if (
                len(row) == 1
                and row[0]
                and not any(
                    char.isdigit() or char in ["+", "-", "%"] for char in row[0]
                )
            ):
                current_section = row[0]
                continue

            # Erstelle Embedding-Text für Datenzeile
            if headers and len(row) == len(headers):
                # Formatiere als "Label -> Header1 value1, Header2 value2"
                label = row[0] if row else ""
                values = []
                for i in range(1, len(row)):
                    if i < len(headers):
                        values.append(f"{headers[i]} {row[i]}")

                embedding_text = f"Table: {header_str}\n"
                if current_section:
                    embedding_text += f"Section: {current_section}\n"
                else:
                    embedding_text += "Section: (none)\n"

                if values:
                    embedding_text += f"Row: {label} -> {', '.join(values)}"
                else:
                    embedding_text += f"Row: {' | '.join(row)}"

                if caption:
                    embedding_text += f"\nCaption: {caption}"

                embedding_text += f"\nSource: page {page} table {table_index}"

                embeddings_texts.append(embedding_text)

        return embeddings_texts

    def embed_tables_and_save_to_duckdb(self, doc_id: int = None) -> dict:
        """
        Lädt Tabellen aus silver_tables, erstellt Embeddings und speichert sie

        Args:
            doc_id: Optional - filtert nach spezifischem Dokument

        Returns:
            Dictionary mit Statistiken (inserted, skipped, total)
        """
        # Lade Tabellen aus DuckLake
        tables = self.load_tables_from_silver(doc_id)

        if not tables:
            print(f"[{datetime.now()}] WARNUNG: Keine Tabellen zum Verarbeiten")
            return {"inserted": 0, "skipped": 0, "total": 0}

        try:
            total_inserted = 0
            total_skipped = 0
            total_embeddings = 0

            for table in tables:
                table_data = table.get("table_data", {})

                # Formatiere Tabelle für Embeddings (eine Zeile pro Embedding)
                embedding_texts = self.format_table_for_embedding(table_data)

                if not embedding_texts:
                    continue

                # Erstelle Embeddings für diese Tabelle
                print(
                    f"[{datetime.now()}] Erstelle {len(embedding_texts)} Embeddings für Tabelle {table['table_id']}..."
                )
                embeddings = self.get_embeddings(embedding_texts)

                # Speichere in Datenbank
                for row_idx, (text, embedding) in enumerate(
                    zip(embedding_texts, embeddings)
                ):
                    try:
                        self.conn.execute(
                            f"""
                            INSERT INTO {self.schema_name}.table_embeddings 
                            (table_id, doc_id, page, table_index, row_index, 
                             embedding_text, embedding, model_name)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                table["table_id"],
                                table["doc_id"],
                                table["page"],
                                table["table_index"],
                                row_idx,
                                text,
                                embedding,  # embedding ist bereits eine Liste
                                self.model_name,
                            ),
                        )
                        total_inserted += 1
                    except Exception as e:
                        total_skipped += 1

                total_embeddings += len(embedding_texts)

            print(
                f"[{datetime.now()}] Tabellen-Embeddings: {total_inserted} neu, {total_skipped} übersprungen"
            )

            return {
                "inserted": total_inserted,
                "skipped": total_skipped,
                "total": total_embeddings,
            }

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER beim Tabellen-Embedding: {e}")
            raise

    def load_chunked_text(self, file_path: str) -> List[str]:
        """
        LEGACY: Lädt bereits gechunkte Text-Daten aus einer Datei
        Für DuckLake verwenden Sie stattdessen load_chunks_from_gold()

        Unterstützt verschiedene Formate:
        - .txt: Ein Chunk pro Zeile
        - .jsonl: JSON Lines Format mit "text", "chunk" oder "content" Feld

        Args:
            file_path: Pfad zur gechunkten Datei

        Returns:
            Liste von Text-Chunks
        """
        print(
            f"[{datetime.now()}] WARNUNG: load_chunked_text() ist Legacy. Verwenden Sie load_chunks_from_gold()"
        )
        chunks = []

        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Suche nach text/chunk/content Feld
                        chunk_text = (
                            data.get("text")
                            or data.get("chunk")
                            or data.get("content", "")
                        )
                        if chunk_text:
                            chunks.append(chunk_text)
        else:
            # Standard: Ein Chunk pro Zeile (.txt)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Ignoriere leere Zeilen
                        chunks.append(line)

        return chunks

    def get_stats(self) -> dict:
        """
        Gibt Statistiken über gespeicherte Embeddings zurück

        Returns:
            Dictionary mit Chunk-Count, Tabellen-Count und Dokument-Count
        """
        chunk_stats = self.conn.execute(
            f"""
            SELECT COUNT(*) as count, COUNT(DISTINCT doc_id) as docs 
            FROM {self.schema_name}.chunk_embeddings
        """
        ).fetchone()

        table_stats = self.conn.execute(
            f"""
            SELECT COUNT(*) as count, COUNT(DISTINCT table_id) as tables 
            FROM {self.schema_name}.table_embeddings
        """
        ).fetchone()

        return {
            "chunk_embeddings": chunk_stats[0],
            "table_embeddings": table_stats[0],
            "total_embeddings": chunk_stats[0] + table_stats[0],
            "unique_docs": chunk_stats[1],
            "unique_tables": table_stats[1],
        }

    def process_all_embeddings(
        self, chunk_config_id: str = None, doc_id: int = None
    ) -> dict:
        """
        High-Level Funktion: Verarbeitet alle Embeddings (Chunks + Tabellen)

        Diese Funktion orchestriert den kompletten Embedding-Prozess:
        1. Lädt und embeddet alle Chunks aus gold_chunks
        2. Lädt und embeddet alle Tabellen aus silver_tables
        3. Gibt kombinierte Statistiken zurück

        Args:
            chunk_config_id: Optional - filtert Chunks nach spezifischer Konfiguration
            doc_id: Optional - filtert nach spezifischem Dokument (Chunks + Tabellen)

        Returns:
            Dictionary mit detaillierten Statistiken für Chunks und Tabellen
        """
        print(f"\n{'='*60}")
        print(f"EMBEDDING-VERARBEITUNG GESTARTET")
        print(f"{'='*60}\n")

        # Schritt 1: Chunks embedden
        print(f"[{datetime.now()}] Schritt 1/2: Verarbeite Chunks aus gold_chunks...")
        chunk_results = self.embed_chunks_and_save_to_duckdb(chunk_config_id, doc_id)
        """
        # Schritt 2: Tabellen embedden
        print(f"\n[{datetime.now()}] Schritt 2/2: Verarbeite Tabellen aus silver_tables...")
        table_results = self.embed_tables_and_save_to_duckdb(doc_id)
        """
        # Kombinierte Statistiken
        total_inserted = chunk_results["inserted"]  # + table_results["inserted"]
        total_skipped = chunk_results["skipped"]  # + table_results["skipped"]
        total_processed = chunk_results["total"]  # + table_results["total"]

        print(f"\n{'='*60}")
        print(f"EMBEDDING-VERARBEITUNG ABGESCHLOSSEN")
        print(f"{'='*60}")
        print(
            f"Chunks:   {chunk_results['inserted']} neu, {chunk_results['skipped']} übersprungen ({chunk_results['total']} gesamt)"
        )
        # print(f"Tabellen: {table_results['inserted']} neu, {table_results['skipped']} übersprungen ({table_results['total']} gesamt)")
        print(
            f"Total:    {total_inserted} neu, {total_skipped} übersprungen ({total_processed} gesamt)"
        )
        print(f"{'='*60}\n")

        # Hole aktuelle Datenbankstatistiken
        db_stats = self.get_stats()

        return {
            "chunks": chunk_results,
            # "tables": table_results,
            "summary": {
                "total_inserted": total_inserted,
                "total_skipped": total_skipped,
                "total_processed": total_processed,
            },
            "database_stats": db_stats,
        }

    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.conn:
            self.conn.close()
            print(f"[{datetime.now()}] Datenbankverbindung geschlossen")

    # ============================================================================
    # LISA'S BEREICH: RAG Prompt-Building und Similarity Search
    # ============================================================================

    def similarity_search(self, user_input: str, k: int = 2) -> List[dict]:
        """
        Führt Similarity Search in chunk_embeddings durch

        Args:
            user_input: Die Frage/Anfrage des Users
            k: Anzahl der Top-K ähnlichsten Chunks

        Returns:
            Liste von Chunk-Dictionaries mit Similarity-Scores
        """
        try:
            # Erstelle Embedding für die User-Query
            print(f"[{datetime.now()}] Erstelle Embedding für Query: {user_input}...")
            query_embedding = self.get_embeddings([user_input])[0]
            print(f"[{datetime.now()}] Query-Embedding erstellt")

            # Führe Similarity Search mit DuckDB VSS durch
            # Verwende array_cosine_similarity für Cosine-Ähnlichkeit
            results = self.conn.execute(
                f"""
                SELECT 
                    embedding_id,
                    chunk_id,
                    doc_id,
                    embedding_text,
                    array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity_score,
                    chunk_config_id,
                    model_name
                FROM {self.schema_name}.chunk_embeddings
                ORDER BY similarity_score DESC
                LIMIT ?
            """,
                [query_embedding, k],
            ).fetchall()

            # Konvertiere zu Dictionary-Liste
            columns = [
                "embedding_id",
                "chunk_id",
                "doc_id",
                "embedding_text",
                "similarity_score",
                "chunk_config_id",
                "model_name",
            ]

            retrieved_docs = []
            for row in results:
                doc = dict(zip(columns, row))
                retrieved_docs.append(doc)

            print(f"[{datetime.now()}] {len(retrieved_docs)} ähnliche Chunks gefunden")
            for i, doc in enumerate(retrieved_docs):
                print(
                    f"  [{i+1}] Similarity: {doc['similarity_score']:.4f} - Chunk ID: {doc['chunk_id']}"
                )
            return retrieved_docs

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei Similarity Search: {e}")
            raise

    def build_prompt_with_context(self, user_input, k: int = 2) -> str:
        """ "
        Args:
            user_input: Die Frage/Anfrage des Users
            k: Anzahl der Top-K ähnlichsten Chunks für Kontext

        Returns:
            Formatierter Prompt mit Kontext für das LLM
        """
        try:
            # Führe Similarity Search durch
            retrieved_docs = self.similarity_search(user_input, k=k)

            if not retrieved_docs:
                print(f"[{datetime.now()}] WARNUNG: Keine relevanten Chunks gefunden")
                return f"Question: {user_input}\n\nI don't have enough context to answer this question."

            # Baue Kontext aus den Top-K Chunks
            retrieved_context = "\n".join(
                [
                    f"[Chunk {i+1} - Similarity: {doc['similarity_score']:.4f}]\n{doc['embedding_text']}"
                    for i, doc in enumerate(retrieved_docs)
                ]
            )

            # Erstelle augmented prompt
            augmented_prompt = f"""Given the context below answer the question.\n
Question: {user_input}\n 
Context: \n{retrieved_context}\n
Remember to answer only based on the context provided and not from any other source.\n
If the question cannot be answered based on the provided context, say I don't know.
"""

            print(
                f"[{datetime.now()}] Prompt mit {len(retrieved_docs)} Kontext-Chunks erstellt"
            )
            return augmented_prompt

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER beim Prompt-Building: {e}")
            raise

    def query_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Sendet Prompt an lokalen LLM-Service

        Args:
            prompt: Der augmented prompt mit Kontext
            temperature: Temperatur für Response-Generierung

        Returns:
            LLM Response als String
        """
        try:
            print(
                f"[{datetime.now()}] Sende Query an LLM-Service (Modell: {self.llm_model_name})..."
            )

            response = requests.post(
                f"{self.llm_service_url}/query",
                json={"prompt": prompt, "temperature": temperature},
                timeout=120,  # 2 Minuten Timeout
            )
            response.raise_for_status()
            result = response.json()

            llm_response = result.get("response", "")
            print(
                f"[{datetime.now()}] LLM Response erhalten ({len(llm_response)} Zeichen)"
            )

            return llm_response

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei LLM Query: {e}")
            raise

    def answer_query(
        self, user_input: str, k: int = 2, temperature: float = 0.0
    ) -> dict:
        """
        High-Level Funktion: Kompletter RAG-Pipeline-Durchlauf

        1. Similarity Search
        2. Prompt Building mit Kontext
        3. LLM Query
        4. Response

        Args:
            user_input: Die Frage/Anfrage des Users
            k: Anzahl der Top-K ähnlichsten Chunks
            model: lokales LLM Modell
            temperature: LLM Temperature

        Returns:
            Dictionary mit Query, Kontext, Prompt und Response
        """
        print(f"\n{'='*60}")
        print(f"RAG QUERY GESTARTET")
        print(f"{'='*60}")
        print(f"Query: {user_input}")
        print(f"Top-K: {k}")
        print(f"Model: {self.llm_model_name}")
        print(f"{'='*60}\n")

        # 1. Hole relevante Chunks
        retrieved_docs = self.similarity_search(user_input, k=k)
        # 2. Baue Prompt
        augmented_prompt = self.build_prompt_with_context(user_input, k=k)
        # 3. Query LLM
        llm_response = self.query_llm(augmented_prompt, temperature=temperature)

        print(f"\n{'='*60}")
        print(f"RAG QUERY ABGESCHLOSSEN")
        print(f"{'='*60}\n")

        return {
            "query": user_input,
            "retrieved_chunks": retrieved_docs,
            "augmented_prompt": augmented_prompt,
            "response": llm_response,
            "model": self.llm_model_name,
            "k": k,
        }
