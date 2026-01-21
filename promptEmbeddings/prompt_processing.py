import os
import duckdb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DuckDB
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import textwrap
from typing import List, Optional

from prompt_loader import PromptLoader


class RAGPipeline:
    """RAG Pipeline: Query → Embedding → Similarity Search → LLM"""
    
    def __init__(
        self, 
        db_path: str, 
        prompts_dir: str = "prompts",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        default_prompt: str = "basic_rag",
        llm_type: str = "ollama",  # "ollama", "openai", "gemini", "huggingface"
        model_name: str = "llama3.2"
    ):
        """
        Initialisiert die RAG Pipeline
        
        Args:
            db_path: Pfad zur DuckDB-Datei mit Embeddings
            prompts_dir: Verzeichnis mit Prompt-Dateien
            embedding_model: Name des HuggingFace Embedding-Modells
            default_prompt: Name des Standard-Prompts
            llm_type: Typ des LLM ("ollama", "openai", "gemini", "huggingface")
            model_name: Name des Modells
        """
        load_dotenv()
        
        self.prompt_loader = PromptLoader(prompts_dir)
        self.default_prompt = default_prompt
        
        # Embedding-Modell initialisieren
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # DuckDB Vector Store verbinden
        self.conn = duckdb.connect(db_path)
        self.vector_store = DuckDB(
            connection=self.conn,
            embedding=self.embeddings
        )
        
        # LLM initialisieren basierend auf Typ
        self.llm = self._init_llm(llm_type, model_name)
    
    def _init_llm(self, llm_type: str, model_name: str):
        """Initialisiert das LLM basierend auf Typ"""
        
        if llm_type == "ollama":
            # Ollama - lokal, kostenlos
            return Ollama(model=model_name)
        
        elif llm_type == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY nicht gefunden in .env")
            return ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=api_key
            )
        
        elif llm_type == "gemini":
            import google.generativeai as genai
            from langchain_core.runnables import Runnable
            
            api_key = os.getenv("GEMINI")
            if not api_key:
                raise ValueError("GEMINI API Key nicht gefunden in .env")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            class GeminiRunnable(Runnable):
                def invoke(self, input, config=None):
                    prompt_str = str(input)
                    response = model.generate_content(prompt_str)
                    return response.text
            
            return GeminiRunnable()
        
        elif llm_type == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint
            api_key = os.getenv("HF")
            if not api_key:
                raise ValueError("HF API Key nicht gefunden in .env")
            return HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=api_key
            )
        
        else:
            raise ValueError(f"Unbekannter LLM-Typ: {llm_type}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> tuple[list, str]:
        """Führt Similarity Search durch und gibt relevante Dokumente zurück"""
        retrieved_docs = self.vector_store.similarity_search(query, k=top_k)
        retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return retrieved_docs, retrieved_context
    
    def query(
        self, 
        question: str, 
        top_k: int = 3, 
        prompt_name: Optional[str] = None,
        verbose: bool = False
    ) -> dict:
        """Vollständiger RAG-Workflow"""
        # 1. Similarity Search
        retrieved_docs, context = self.retrieve_context(question, top_k)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"QUERY: {question}")
            print(f"{'='*80}")
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                preview = textwrap.shorten(doc.page_content, 100)
                print(f"  {i}. {preview}")
        
        # 2. Prompt erstellen
        template_name = prompt_name or self.default_prompt
        augmented_prompt = self.prompt_loader.format_prompt(
            name=template_name,
            query=question,
            context=context
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PROMPT TEMPLATE: {template_name}")
            print(f"{'='*80}")
            print(textwrap.fill(augmented_prompt, width=100))
        
        # 3. LLM-Antwort generieren
        try:
            response = self.llm.invoke(augmented_prompt)
            # Für ChatOpenAI
            if hasattr(response, 'content'):
                response = response.content
        except Exception as e:
            response = f"Fehler beim LLM-Aufruf: {e}"
        
        if verbose:
            print(f"\n{'='*80}")
            print("ANTWORT:")
            print(f"{'='*80}")
            print(textwrap.fill(str(response), width=100))
        
        return {
            "question": question,
            "answer": str(response),
            "context": context,
            "retrieved_docs": retrieved_docs,
            "num_docs": len(retrieved_docs),
            "prompt_template": template_name
        }
    
    def list_prompts(self) -> List[str]:
        """Gibt eine Liste aller verfügbaren Prompts zurück"""
        return self.prompt_loader.list_prompts()
    
    def show_prompt_info(self):
        """Zeigt Informationen über alle geladenen Prompts"""
        self.prompt_loader.print_prompt_info()
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Beispiel-Verwendung mit Ollama"""
    
    db_path = "db/vector.duckdb"
    
    # Mit Ollama (lokal, kostenlos)
    with RAGPipeline(
        db_path, 
        prompts_dir="prompts",
        llm_type="ollama",
        model_name="llama3.2" 
    ) as rag:
        
        rag.show_prompt_info()
        
        query = "Which pipelines are used in RAG?"
        
        result = rag.query(
            question=query,
            top_k=2,
            prompt_name="basic_rag",
            verbose=True
        )


if __name__ == "__main__":
    main()