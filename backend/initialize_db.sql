CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;


CREATE TABLE IF NOT EXISTS bronze.documents (
            doc_id VARCHAR PRIMARY KEY,
            doc_name VARCHAR NOT NULL,
            source_type VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            content_hash VARCHAR NOT NULL,
            file_size_bytes BIGINT,
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        );

CREATE TABLE IF NOT EXISTS silver.document_text (
    doc_id VARCHAR PRIMARY KEY,
    extracted_text TEXT,
    cleaned_text TEXT,
    extraction_method VARCHAR,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS silver.document_chunks (
    chunk_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    section_title VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gold.retrieval_chunks (
    chunk_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    document_name VARCHAR,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    embedding FLOAT[384] NOT NULL,
    embedding_model VARCHAR NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SET hnsw_enable_experimental_persistence=true;
CREATE INDEX IF NOT EXISTS gold_chunk_embeddings_hnsw
        ON gold.retrieval_chunks
        USING HNSW (embedding);
