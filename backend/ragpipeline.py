import os
import duckdb
from typing import List
from datetime import datetime
import requests
import time
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstrakte Basis-Klasse für LLM-Provider"""
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Gibt den Namen des aktuellen Modells zurück"""
        pass
    
    @abstractmethod
    def query(self, prompt: str, temperature: float = 0.0) -> str:
        """Sendet einen Prompt an den LLM und gibt die Response zurück"""
        pass


class LocalLLMProvider(LLMProvider):
    """Provider für lokale LLM-Services"""
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.model_name = "unknown"
        self._fetch_model_info()
    
    def _fetch_model_info(self):
        """Holt Modell-Informationen vom lokalen LLM-Service"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.service_url}/model-info", timeout=30)
                response.raise_for_status()
                info = response.json()
                self.model_name = info.get("model_name", "unknown")
                print(f"[{datetime.now()}] Local LLM-Service verbunden: {self.model_name}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{datetime.now()}] WARNUNG: Kann Local LLM Model-Info nicht abrufen (Versuch {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"[{datetime.now()}] FEHLER: Kann Local LLM Model-Info nicht abrufen nach {max_retries} Versuchen")
                    self.model_name = "unknown"
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def query(self, prompt: str, temperature: float = 0.0) -> str:
        """Sendet Prompt an lokalen LLM-Service"""
        try:
            response = requests.post(
                f"{self.service_url}/query",
                json={"prompt": prompt, "temperature": temperature},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei Local LLM Query: {e}")
            raise


class OpenAILLMProvider(LLMProvider):
    """Provider für OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        
        if not self.api_key:
            raise ValueError("OpenAI API Key nicht gefunden. Setze OPENAI_API_KEY Umgebungsvariable.")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        print(f"[{datetime.now()}] OpenAI Provider initialisiert mit Modell: {self.model}")
    
    def get_model_name(self) -> str:
        return self.model
    
    def query(self, prompt: str, temperature: float = 0.0) -> str:
        """Sendet Prompt an OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=120,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei OpenAI Query: {e}")
            raise


def get_llm_provider() -> LLMProvider:
    """Factory-Funktion um den passenden LLM-Provider zu erstellen"""
    llm_type = os.getenv("LLM_TYPE", "local").lower()
    
    if llm_type == "openai":
        return OpenAILLMProvider()
    else:
        # Default: Local LLM
        llm_service_url = os.getenv("LLM_SERVICE_URL", "http://llm-service:11434")
        return LocalLLMProvider(llm_service_url)


class RagPipeline:

    def __init__(
        self,
        db_path: str = os.getenv("DUCKDB_PATH", "/app/DuckLake/ducklake.duckdb"),
        embedding_service_url: str = None,
        schema_name: str = None,
        lazy_init: bool = True
    ):
        """
        Initialisiert die RAG Pipeline für Query & Retrieval
        
        Args:
            db_path: Pfad zur DuckDB Datenbank
            embedding_service_url: URL des Embedding-Service
            schema_name: DuckDB Schema für Embeddings (default: aus ENV oder 'embeddings_e5_small')
            lazy_init: Wenn True, wird DB erst bei erstem Zugriff geöffnet
        """
        self.db_path = db_path
        self.embedding_service_url = embedding_service_url or os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
        self.schema_name = schema_name or os.getenv("EMBEDDINGS_SCHEMA", "embeddings_e5_small")
        self.embedding_dim = None
        self.model_name = None
        self.conn = None
        self._lazy_init = lazy_init
        
        # Initialisiere LLM-Provider (flexibel: local oder OpenAI)
        self.llm_provider = get_llm_provider()

        # Initialisiere bei Instanziierung
        self._fetch_model_info()

        if not lazy_init:
            self._initialize_db()
    
    def _fetch_model_info(self):
        """Holt Modell-Informationen vom Embedding-Service und setzt embedding_dim"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.embedding_service_url}/model-info", timeout=30)
                response.raise_for_status()
                info = response.json()
                
                self.embedding_dim = info["dimension"]
                self.model_name = info["model_name"]
                
                print(f"[{datetime.now()}] Embedding-Service verbunden:")
                print(f"  - Modell: {self.model_name}")
                print(f"  - Dimension: {self.embedding_dim}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{datetime.now()}] WARNUNG: Kann Modell-Info nicht abrufen (Versuch {attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"[{datetime.now()}] FEHLER: Kann Modell-Info nicht abrufen nach {max_retries} Versuchen: {e}")
                    print(f"[{datetime.now()}] Verwende Standard-Dimension: 384")
                    self.embedding_dim = 384  # Größe für e5-small
                    self.model_name = "unknown"
    
    def _fetch_llm_model_info(self):
        """Diese Methode wird nicht mehr benötigt - Info kommt vom Provider"""
        pass
    
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
                timeout=300  # 5 Minuten Timeout für große Batches
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
        
        # Öffne zunächst im Read-Write Modus, um Verbindung zu etablieren
        # Die DB wird von Ingestion-Service erstellt und mit Daten gefüllt
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            print(f"[{datetime.now()}] DuckDB im Read-Only Modus geöffnet")
        except Exception as e:
            print(f"[{datetime.now()}] WARNUNG: Kann DuckDB im Read-Only nicht öffnen: {e}")
            print(f"[{datetime.now()}] Öffne im Read-Write Modus...")
            self.conn = duckdb.connect(self.db_path, read_only=False)
            print(f"[{datetime.now()}] DuckDB im Read-Write Modus geöffnet")

    def _ensure_connection(self):
        """Stellt sicher, dass eine DB-Verbindung besteht (lazy loading)"""
        if self.conn is None:
            self._initialize_db()


    def similarity_search(self, user_input: str, k: int = 2) -> List[dict]:
        """
        Führt Similarity Search in chunk_embeddings durch
        
        Args:
            user_input: Die Frage/Anfrage des Users
            k: Anzahl der Top-K ähnlichsten Chunks
            
        Returns:
            Liste von Chunk-Dictionaries mit Similarity-Scores
        """
        self._ensure_connection()

        try:
            # Erstelle Embedding für die User-Query
            print(f"[{datetime.now()}] Erstelle Embedding für Query: {user_input}...")
            query_embedding = self.get_embeddings([user_input])[0]
            print(f"[{datetime.now()}] Query-Embedding erstellt (Größe: {len(query_embedding)})")
            print(f"[{datetime.now()}] Expected Dimension: {self.embedding_dim}")
            
            # Konvertiere Query-Embedding zu SQL-Array
            query_embedding_sql = str(query_embedding)
            
            # Führe Similarity Search mit DuckDB VSS durch
            # Verwende array_cosine_similarity für Cosine-Ähnlichkeit
            results = self.conn.execute(f"""
                SELECT 
                    embedding_id,
                    chunk_id,
                    doc_id,
                    embedding_text,
                    array_cosine_similarity(embedding, {query_embedding_sql}::FLOAT[{self.embedding_dim}]) as similarity_score,
                    chunk_config_id,
                    model_name
                FROM gold_embeddings
                ORDER BY similarity_score DESC
                LIMIT {k}
            """).fetchall()
            
            # Konvertiere zu Dictionary-Liste
            columns = ['embedding_id', 'chunk_id', 'doc_id', 'embedding_text', 
                      'similarity_score', 'chunk_config_id', 'model_name']
            
            retrieved_docs = []
            for row in results:
                doc = dict(zip(columns, row))
                retrieved_docs.append(doc)
            
            print(f"[{datetime.now()}] {len(retrieved_docs)} ähnliche Chunks gefunden")
            for i, doc in enumerate(retrieved_docs):
                print(f"  [{i+1}] Similarity: {doc['similarity_score']:.4f} - Chunk ID: {doc['chunk_id']}")
            return retrieved_docs
            
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei Similarity Search: {e}")
            print(f"[{datetime.now()}] Embedding Dimension: {self.embedding_dim}, Query Embedding Size: {len(query_embedding)}")
            raise
        
    def build_prompt_with_context(self, user_input, k: int = 2) -> str:

        """"
        Args:
            user_input: Die Frage/Anfrage des Users
            k: Anzahl der Top-K ähnlichsten Chunks für Kontext
            
        Returns:
            Formatierter Prompt mit Kontext für das LLM
        """
        self._ensure_connection()

        try: 
            # Führe Similarity Search durch
            retrieved_docs = self.similarity_search(user_input, k=k)
            
            if not retrieved_docs:
                print(f"[{datetime.now()}] WARNUNG: Keine relevanten Chunks gefunden")
                return f"Question: {user_input}\n\nI don't have enough context to answer this question."
            
            # Baue Kontext aus den Top-K Chunks
            retrieved_context = "\n".join([
                f"[Chunk {i+1} - Similarity: {doc['similarity_score']:.4f}]\n{doc['embedding_text']}"
                for i, doc in enumerate(retrieved_docs)
            ])

            # Erstelle augmented prompt
            augmented_prompt = f"""Given the context below answer the question.\n
Question: {user_input}\n 
Context: \n{retrieved_context}\n
Remember to answer only based on the context provided and not from any other source.\n
If the question cannot be answered based on the provided context, say I don't know.
"""
                        
            print(f"[{datetime.now()}] Prompt mit {len(retrieved_docs)} Kontext-Chunks erstellt")
            return augmented_prompt

        except Exception as e:
            print(f"[{datetime.now()}] FEHLER beim Prompt-Building: {e}")
            raise

    def query_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Sendet Prompt an LLM (lokal oder OpenAI)
        
        Args:
            prompt: Der augmented prompt mit Kontext
            temperature: Temperatur für Response-Generierung
            
        Returns:
            LLM Response als String
        """
        try:
            print(f"[{datetime.now()}] Sende Query an LLM-Provider (Modell: {self.llm_provider.get_model_name()})...")
            
            llm_response = self.llm_provider.query(prompt, temperature)
            print(f"[{datetime.now()}] LLM Response erhalten ({len(llm_response)} Zeichen)")
        
            return llm_response
        
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER bei LLM Query: {e}")
            raise
    
    def answer_query(self, user_input: str, k: int = 2, temperature: float = 0.0) -> dict:
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
        try:
            print(f"\n{'='*60}")
            print(f"RAG QUERY GESTARTET")
            print(f"{'='*60}")
            print(f"Query: {user_input}")
            print(f"Top-K: {k}")
            print(f"Model: {self.llm_provider.get_model_name()}")
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
                "model": self.llm_provider.get_model_name(),
                "k": k
            }
        except Exception as e:
            print(f"[{datetime.now()}] FEHLER in answer_query: {e}")
            raise

    def get_stats(self) -> dict:
        """
        Gibt Statistiken über gespeicherte Embeddings zurück
        
        Returns:
            Dictionary mit Chunk-Count, Tabellen-Count und Dokument-Count
        """
        self._ensure_connection()
        
        chunk_stats = self.conn.execute(
            """
            SELECT COUNT(*) as count, COUNT(DISTINCT doc_id) as docs 
            FROM gold_embeddings
            """
        ).fetchone()
        
        return {
            "chunk_embeddings": chunk_stats[0],
            "table_embeddings": 0,
            "total_embeddings": chunk_stats[0],
            "unique_docs": chunk_stats[1],
            "unique_tables": 0
        }
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.conn:
            self.conn.close()
            print(f"[{datetime.now()}] Datenbankverbindung geschlossen")
