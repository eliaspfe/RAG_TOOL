"""
Test-Script für die RAG Pipeline
Testet die Embedding-Funktionalität mit Mock-Daten
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ragpipeline import RagPipeline
from datetime import datetime

def setup_test_data(pipeline):
    """Erstellt Test-Embeddings in der Datenbank"""
    print("\n" + "="*60)
    print("SETUP: Test-Daten erstellen")
    print("="*60)
    
    test_chunks = [
        "RAG steht für Retrieval-Augmented Generation. Es kombiniert Information Retrieval mit Text-Generierung.",
        "Machine Learning Pipelines bestehen aus mehreren Stufen: Data Collection, Preprocessing, Training und Evaluation.",
        "Vector Databases speichern Embeddings und ermöglichen effiziente Similarity Search mit Cosine Similarity."
    ]
    
    print(f"Erstelle Embeddings für {len(test_chunks)} Test-Chunks...")
    embeddings = pipeline.get_embeddings(test_chunks)
    
    # Speichere in DB
    for i, (text, embedding) in enumerate(zip(test_chunks, embeddings)):
        try:
            pipeline.conn.execute(f"""
                INSERT INTO {pipeline.schema_name}.chunk_embeddings 
                (chunk_id, chunk_row_id, doc_id, embedding_text, embedding, 
                 embedding_type, chunk_config_id, run_id, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f'test_chunk_{i+1}',
                i+1,
                1,
                text,
                embedding,
                'text',
                'test_config',
                'test_run_001',
                pipeline.model_name
            ))
        except:
            pass  # Chunk bereits vorhanden
    
    print(f"✅ Test-Daten erstellt")



def test_similarity_search(pipeline):
    """Testet die Similarity Search Funktion"""
    print("\n" + "="*60)
    print("TEST 1: Similarity Search")
    print("="*60)
    
    query = "Was ist RAG?"
    print(f"Query: '{query}'")
    
    results = pipeline.similarity_search(query, k=2)
    
    print(f"✅ {len(results)} Ergebnisse gefunden:")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] Similarity: {doc['similarity_score']:.4f}")
        print(f"      Text: {doc['embedding_text'][:80]}...")

def test_build_prompt(pipeline):
    """Testet das Prompt-Building"""
    print("\n" + "="*60)
    print("TEST 2: Prompt Building")
    print("="*60)
    
    query = "Erkläre mir Machine Learning Pipelines"    
    prompt = pipeline.build_prompt_with_context(query, k=2)
    
    print(f"✅ Prompt erstellt ({len(prompt)} Zeichen)")
    print("\n" + "-"*60)
    print(prompt[:400] + "...")
    print("-"*60)

def test_llm_query(pipeline):
    """Testet die LLM Query"""
    print("\n" + "="*60)
    print("TEST 3: LLM Query")
    print("="*60)
    
    test_prompt = "Erkläre in einem Satz was Machine Learning ist."
    print(f"Test-Prompt: '{test_prompt}'")
    
    try:
        response = pipeline.query_llm(test_prompt, temperature=0.0)
        
        print(f"✅ LLM Response erhalten ({len(response)} Zeichen):")
        print("\n" + "-"*60)
        print(response)
        print("-"*60)
    except Exception as e:
        print(f"⚠️  LLM-Service nicht verfügbar: {e}")


def test_full_pipeline(pipeline):
    """Testet die komplette RAG-Pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Vollständige RAG-Pipeline")
    print("="*60)
    
    query = "Was ist Vector Database und Similarity Search?"
    print(f"Query: '{query}'")
    
    try:
        result = pipeline.answer_query(query, k=2, temperature=0.0)
        
        print(f"✅ Pipeline abgeschlossen!")
        print(f"\n📋 Retrieved Chunks: {len(result['retrieved_chunks'])}")
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"  [{i+1}] Score: {chunk['similarity_score']:.4f}")
        
        print(f"\n🤖 LLM Response:")
        print("-"*60)
        print(result['response'])
        print("-"*60)
    except Exception as e:
        print(f"⚠️  Pipeline-Fehler: {e}")

def cleanup(pipeline):
    """Räumt Test-Datenbank auf"""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    if pipeline:
        try:
            pipeline.conn.execute(f"""
                DELETE FROM {pipeline.schema_name}.chunk_embeddings 
                WHERE chunk_config_id = 'test_config'
            """)
            print("✅ Test-Daten gelöscht")
        except:
            pass
        
        pipeline.close()
        print("✅ Datenbankverbindung geschlossen")
    
    import shutil
    if os.path.exists("test_db"):
        shutil.rmtree("test_db")
        print("✅ Test-Datenbank gelöscht")

def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("RAG QUERY TEST SUITE")
    print(f"Gestartet: {datetime.now()}")
    print("="*60)
    
    try:
        pipeline = RagPipeline(
            db_path="test_db/vector.duckdb",
            embedding_service_url="http://localhost:8001",
            llm_service_url="http://localhost:11434",
            schema_name="test_RAG"
        )
        
        setup_test_data(pipeline)
        test_similarity_search(pipeline)
        test_build_prompt(pipeline)
        test_llm_query(pipeline)
        test_full_pipeline(pipeline)
        cleanup(pipeline)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
    
    print("\n" + "="*60)
    print("TESTS ABGESCHLOSSEN")
    print("="*60)
    print("\nHinweise:")
    print("  docker-compose up embedding-service")

if __name__ == "__main__":
    main()