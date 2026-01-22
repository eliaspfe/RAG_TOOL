import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn

app = FastAPI(title="Embedding Service")

# Globale Variable für das Modell
model = None
model_name = None


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.on_event("startup")
async def load_model():
    """Lädt das Embedding-Modell beim Start"""
    global model, model_name
    
    model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-de")
    print(f"Lade Embedding-Modell: {model_name}")
    
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"Modell geladen! Dimension: {model.get_sentence_embedding_dimension()}")


@app.get("/health")
async def health():
    """Health-Check Endpunkt"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": model_name,
        "dimension": model.get_sentence_embedding_dimension()
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    Erstellt Embeddings für die übergebenen Texte
    
    Args:
        request: Liste von Texten
        
    Returns:
        Liste von Embedding-Vektoren
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        # Erstelle Embeddings
        embeddings = model.encode(request.texts, show_progress_bar=False)
        
        # Konvertiere zu Liste von Listen
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        return EmbedResponse(
            embeddings=embeddings_list,
            model=model_name,
            dimension=len(embeddings_list[0])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Gibt Informationen über das geladene Modell zurück"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
