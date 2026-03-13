import duckdb
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import uuid
from typing import List
import requests


class RagPipeline:
    def __init__(
        self,
        db_path: str = "./data/rag_database.duckdb",
        embedding_service_url: str = "http://127.0.0.1:8001",
    ):
        load_dotenv(override=True)
        self.db_path = db_path
        self.client = OpenAI()
        self.conn = duckdb.connect(self.db_path)
        if os.getenv("EMBEDDING_TYPE") == "api":
            self.client = OpenAI()

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

    def similarity_search(self, query: str, top_k: int = 5):
        # Query embedding erzeugen (Dimension = 384)
        query_embedding = self.get_embeddings([query])[0]

        results = self.conn.execute(
            f"""
            SELECT
                chunk_id,
                doc_id,
                chunk_text,
                array_cosine_similarity(
                    embedding,
                    CAST(? AS FLOAT[384])
                ) AS similarity,
                document_name,
                page_number
            FROM gold.retrieval_chunks
            ORDER BY similarity DESC
            LIMIT ?
            """,
            (query_embedding, top_k),
        ).fetchall()

        return [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "chunk_text": row[2],
                "similarity": row[3],
                "doc_name": row[4],
                "page_number": row[5],
            }
            for row in results
        ]

    def build_prompt_from_chunks(self, query: str, similar_chunks: List[dict]):
        context_texts = [row["chunk_text"] for row in similar_chunks]

        context = "\n\n".join(
            [f"Kontext {i+1}:\n{text}" for i, text in enumerate(context_texts)]
        )

        prompt = f"""Beantworte jede Frage des Nutzers mit Hilfe des hier übegebenen Kontextes. Sollte die Antwort nicht zu finden sein weise den Nutzer darauf hin, dass die Informationen nicht gefunden werden konnten.
    Gesamter Kontext:
    {context}

    Nutzer Frage:
    {query}
    """

        return prompt

    def build_prompt(self, query: str, top_k: int = 5):
        """
        Erstellt einen Prompt mit Kontext aus den ähnlichsten Chunks.

        Args:
            query: Nutzerfrage
            top_k: Anzahl ähnlicher Chunks

        Returns:
            Prompt-String
        """

        similar_chunks = self.similarity_search(query, top_k=top_k)
        return self.build_prompt_from_chunks(query, similar_chunks)
