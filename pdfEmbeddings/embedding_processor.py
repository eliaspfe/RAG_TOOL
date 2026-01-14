import os
import duckdb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from datetime import datetime
import time
import json

class TextEmbeddingProcessor:
    """
    Verarbeitet bereits gechunkte Text-Dokumente und speichert sie als Vektoren in DuckDB
    """
    
    def __init__(self, db_path: str, model_name: str = "jinaai/jina-embeddings-v2-base-de", embedding_dim: int = 768):
        """
        Initialisiert den Processor
        
        Args:
            db_path: Pfad zur DuckDB Datenbank
            model_name: Name des Embedding-Modells
            embedding_dim: Dimension der Embeddings (768 für Jina v2)
        """
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.conn = None
        
    def log(self, level: str, message: str, details: dict = None):
        """
        Schreibt Log-Einträge in die Datenbank
        
        Args:
            level: Log-Level (INFO, WARNING, ERROR, DEBUG)
            message: Log-Nachricht
            details: Optionale zusätzliche Details als JSON
        """
        if self.conn:
            try:
                details_json = json.dumps(details) if details else None
                self.conn.execute("""
                    INSERT INTO processing_logs (level, message, details)
                    VALUES (?, ?, ?)
                """, (level, message, details_json))
            except Exception:
                pass  # Fehler beim Logging ignorieren
        
    def initialize(self):
        """Initialisiert das Embedding-Modell und die Datenbank"""
        print(f"[{datetime.now()}] Lade Embedding-Modell: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        print(f"[{datetime.now()}] Modell geladen!")
        print(f"[{datetime.now()}] Verbinde mit DuckDB: {self.db_path}")
        self.conn = duckdb.connect(self.db_path)
        
        # Installiere und lade VSS Extension für Vektor-Ähnlichkeitssuche
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        
        # Erstelle Log-Tabelle
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS processing_logs_seq START 1;
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY DEFAULT nextval('processing_logs_seq'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level VARCHAR,
                message TEXT,
                details TEXT
            )
        """)
        
        # Erstelle Tabelle für Text-Chunks mit Vektoren
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS text_embeddings_seq START 1;
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS text_embeddings (
                id INTEGER PRIMARY KEY DEFAULT nextval('text_embeddings_seq'),
                chunk_text TEXT,
                embedding FLOAT[{self.embedding_dim}],
                source_file TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_file, chunk_index)
            )
        """)
        
        self.log("INFO", "System initialisiert", {
            "model": self.model_name,
            "db_path": self.db_path
        })
        print(f"[{datetime.now()}] System erfolgreich initialisiert!")
        
    def load_chunked_text(self, file_path: str) -> List[str]:
        """
        Lädt bereits gechunkte Text-Daten aus einer Datei
        Unterstützt verschiedene Formate:
        - .txt: Ein Chunk pro Zeile
        - .jsonl: JSON Lines Format mit "text" oder "chunk" Feld
        
        Args:
            file_path: Pfad zur gechunkten Datei
            
        Returns:
            Liste von Text-Chunks
        """
        chunks = []
        
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Suche nach text/chunk Feld
                        chunk_text = data.get('text') or data.get('chunk') or data.get('content', '')
                        if chunk_text:
                            chunks.append(chunk_text)
        else:
            # Standard: Ein Chunk pro Zeile
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Ignoriere leere Zeilen
                        chunks.append(line)
        
        return chunks
    
    def process_chunked_file(self, file_path: str):
        """
        Verarbeitet eine bereits gechunkte Text-Datei: Embeddings erstellen und speichern
        
        Args:
            file_path: Pfad zur gechunkten Text-Datei
        """
        try:
            # Lade bereits gechunkte Texte
            print(f"[{datetime.now()}] Lade Chunks aus: {os.path.basename(file_path)}")
            chunks = self.load_chunked_text(file_path)
            print(f"[{datetime.now()}] {len(chunks)} Chunks geladen")
            self.log("INFO", "Datei geladen", {
                "file": os.path.basename(file_path),
                "chunks": len(chunks)
            })
            
            if not chunks:
                print(f"[{datetime.now()}] WARNUNG: Keine Chunks in {os.path.basename(file_path)}")
                self.log("WARNING", "Keine Chunks gefunden", {
                    "file": os.path.basename(file_path)
                })
                return
            
            # Erstelle Embeddings
            print(f"[{datetime.now()}] Erstelle Embeddings...")
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            print(f"[{datetime.now()}] Embeddings erstellt: {len(embeddings)}")
            self.log("DEBUG", "Embeddings erstellt", {
                "file": os.path.basename(file_path),
                "count": len(embeddings)
            })
            
            # Speichere in Datenbank
            print(f"[{datetime.now()}] Speichere in DB...")
            inserted = 0
            skipped = 0
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    self.conn.execute("""
                        INSERT INTO text_embeddings (chunk_text, embedding, source_file, chunk_index)
                        VALUES (?, ?, ?, ?)
                    """, (chunk, embedding.tolist(), os.path.basename(file_path), idx))
                    inserted += 1
                except Exception:
                    skipped += 1
            print(f"[{datetime.now()}] In DB gespeichert: {inserted} neu, {skipped} übersprungen (Duplikate)")
            
            self.log("INFO", "Chunks gespeichert", {
                "file": os.path.basename(file_path),
                "count": len(chunks)
            })
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei {os.path.basename(file_path)}: {e}")
            self.log("ERROR", "Fehler bei Verarbeitung", {
                "file": os.path.basename(file_path),
                "error": str(e)
            })
        
    def process_directory(self, directory: str, file_extensions: List[str] = [".txt", ".jsonl"]):
        """
        Verarbeitet alle gechunkten Dateien in einem Verzeichnis
        
        Args:
            directory: Pfad zum Verzeichnis
            file_extensions: Liste von Dateiendungen zum Verarbeiten
        """
        files = [f for f in os.listdir(directory) 
                if any(f.endswith(ext) for ext in file_extensions)]
        
        print(f"[{datetime.now()}] Gefundene Dateien: {len(files)} - {files}")
        
        self.log("INFO", "Verarbeitung gestartet", {
            "directory": directory,
            "files_found": len(files)
        })
        
        for file in files:
            file_path = os.path.join(directory, file)
            print(f"[{datetime.now()}] Verarbeite: {file}")
            self.process_chunked_file(file_path)
            print(f"[{datetime.now()}] Fertig: {file}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Sucht ähnliche Text-Chunks zu einer Anfrage
        
        Args:
            query: Suchanfrage
            top_k: Anzahl der Ergebnisse
            
        Returns:
            Liste von (Text, Ähnlichkeitsscore) Tupeln
        """
        # Erstelle Embedding für Query
        query_embedding = self.model.encode([query])[0]
        
        # Suche ähnliche Vektoren
        result = self.conn.execute(f"""
            SELECT chunk_text, array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity
            FROM text_embeddings
            ORDER BY similarity DESC
            LIMIT ?
        """, (query_embedding.tolist(), top_k)).fetchall()
        
        return result
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.conn:
            self.log("INFO", "System heruntergefahren", {})
            self.conn.close()


def main():
    """Hauptfunktion"""
    # Konfiguration aus Umgebungsvariablen
    db_path = os.getenv("DUCKDB_PATH", "/app/db/vector.duckdb")
    model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-de")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    data_dir = os.getenv("DATA_DIR", "/app/data")
    
    # Erstelle Verzeichnisse falls nicht vorhanden
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialisiere Processor
    processor = TextEmbeddingProcessor(db_path, model_name, embedding_dim)
    processor.initialize()
    
    print(f"[{datetime.now()}] Prüfe Daten-Verzeichnis: {data_dir}")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"[{datetime.now()}] Starte Verarbeitung...")
        processor.process_directory(data_dir)
    else:
        print(f"[{datetime.now()}] Keine Dateien gefunden in {data_dir}")
        processor.log("WARNING", "Keine Dateien gefunden", {
            "directory": data_dir,
            "expected_formats": [".txt", ".jsonl"]
        })
    
    # Zeige Statistiken
    stats = processor.conn.execute(
        "SELECT COUNT(*) as count, COUNT(DISTINCT source_file) as files FROM text_embeddings"
    ).fetchone()
    print(f"[{datetime.now()}] Verarbeitung abgeschlossen: {stats[0]} Chunks, {stats[1]} Dateien")
    processor.log("INFO", "Verarbeitung abgeschlossen", {
        "total_chunks": stats[0],
        "total_files": stats[1]
    })
    
    processor.close()
    
    # Halte Container am Leben
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
