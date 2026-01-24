import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="LLM Service")

# Global variables for the model and tokenizer
llm = None
tokenizer = None
model_name = None
device = None

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

@app.on_event("startup")
async def load_model():
    """Load the LLM model on startup"""
    global llm, tokenizer, model_name, device

    # Detect device (Apple MPS or CPU)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load model name from environment or default
    model_name = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") 
    print(f"Loading LLM model: {model_name}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model (this may take a while)...")
        llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        print(f"Model {model_name} loaded successfully!")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": model_name, "device": str(device)}

@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    """Query the LLM with a prompt and return the response"""
    if llm is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize and move inputs to device
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = llm.generate(**inputs, max_new_tokens=100)

        # Decode to text
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying LLM: {e}")
    

@app.get("/model-info")
async def model_info():
    """Returns information about the loaded LLM model"""
    if llm is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": llm.config.max_position_embeddings if hasattr(llm.config, "max_position_embeddings") else "Unknown",
        "num_parameters": sum(p.numel() for p in llm.parameters() if p.requires_grad)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)