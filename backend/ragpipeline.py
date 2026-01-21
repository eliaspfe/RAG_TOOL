class RagPipeline:

    def __init__(self):
        # initialize any required attributes here
        # embedding model, database connections, etc.
        # create schema, tables in duckdb if needed
        pass

    def test_func(self):
        return 15

    def ducklake(self, file_path):  # Noah
        return "Daten aus PDFs extrahiert und in DuckDB gespeichert."

    def embed_chunks_and_save_to_duckdb(self):  # Felix
        return "Embedding und Speichern in DuckDB abgeschlossen."

    def build_prompt_with_context(self, user_input) -> str:  # Lisa
        # similarity search in DuckDB und Kontext zurückgeben
        context = "Dies ist ein Beispielkontext aus der DuckDB."
        prompt = (
            "Based on the following context: "
            + context
            + " Answer the following question: "
            + user_input
        )
        return prompt
