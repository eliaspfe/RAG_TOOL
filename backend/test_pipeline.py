"""
Test-Script für die RAG Pipeline
Testet die Embedding-Funktionalität mit Mock-Daten
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ragpipeline import RagPipeline
from datetime import datetime

def test_embedding_service_connection():
    """Testet die Verbindung zum Embedding-Service"""
    print("\n" + "="*60)
    print("TEST 1: Embedding-Service Verbindung")
    print("="*60)
    
    try:
        # Teste mit localhost (für lokalen Test)
        pipeline = RagPipeline(
            db_path="test_db/vector.duckdb",
            embedding_service_url="http://localhost:8001",
            schema_name="test_embeddings"
        )
        print("✅ Verbindung erfolgreich!")
        print(f"   Modell: {pipeline.model_name}")
        print(f"   Dimension: {pipeline.embedding_dim}")
        return pipeline
    except Exception as e:
        print(f"❌ Fehler bei Verbindung: {e}")
        return None

def test_get_embeddings(pipeline):
    """Testet das Abrufen von Embeddings"""
    print("\n" + "="*60)
    print("TEST 2: Embeddings abrufen")
    print("="*60)
    
    if not pipeline:
        print("⚠️  Übersprungen (keine Verbindung)")
        return
    
    try:
        test_texts = [
            "Dies ist ein Test-Satz.",
            "Machine Learning ist faszinierend.",
            "Die Katze sitzt auf der Matte."
        ]
        
        print(f"Erstelle Embeddings für {len(test_texts)} Texte...")
        embeddings = pipeline.get_embeddings(test_texts)
        
        print(f"✅ {len(embeddings)} Embeddings erhalten")
        print(f"   Dimension: {len(embeddings[0])}")
        print(f"   Erste Werte: {embeddings[0][:5]}...")
        
    except Exception as e:
        print(f"❌ Fehler beim Embedding: {e}")

def test_database_operations(pipeline):
    """Testet Datenbank-Operationen"""
    print("\n" + "="*60)
    print("TEST 3: Datenbank-Operationen")
    print("="*60)
    
    if not pipeline:
        print("⚠️  Übersprungen (keine Verbindung)")
        return
    
    try:
        # Teste ob Schema existiert
        result = pipeline.conn.execute(f"""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = '{pipeline.schema_name}'
        """).fetchone()
        
        if result:
            print(f"✅ Schema '{pipeline.schema_name}' existiert")
        
        # Teste ob Tabellen existieren
        tables = pipeline.conn.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{pipeline.schema_name}'
        """).fetchall()
        
        print(f"✅ {len(tables)} Tabellen gefunden:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Teste Statistiken
        stats = pipeline.get_stats()
        print(f"✅ Statistiken:")
        print(f"   - Chunk Embeddings: {stats['chunk_embeddings']}")
        print(f"   - Tabellen Embeddings: {stats['table_embeddings']}")
        
    except Exception as e:
        print(f"❌ Fehler bei DB-Operationen: {e}")

def cleanup(pipeline):
    """Räumt Test-Datenbank auf"""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    if pipeline:
        pipeline.close()
        print("✅ Datenbankverbindung geschlossen")
    
    # Lösche Test-DB
    import shutil
    if os.path.exists("test_db"):
        shutil.rmtree("test_db")
        print("✅ Test-Datenbank gelöscht")

def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("RAG PIPELINE TEST SUITE")
    print(f"Gestartet: {datetime.now()}")
    print("="*60)
    
    pipeline = test_embedding_service_connection()
    test_get_embeddings(pipeline)
    test_database_operations(pipeline)
    cleanup(pipeline)
    
    print("\n" + "="*60)
    print("TESTS ABGESCHLOSSEN")
    print("="*60)
    print("\nHinweis: Stellen Sie sicher, dass der Embedding-Service läuft:")
    print("  docker-compose up embedding-service")

if __name__ == "__main__":
    main()
