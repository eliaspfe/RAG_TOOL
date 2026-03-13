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
max_new_tokens = None


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    response: str


def get_candidate_models() -> list[str]:
    primary_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip()
    fallback_raw = os.getenv("LLM_FALLBACK_MODELS", "Qwen/Qwen2.5-0.5B-Instruct")
    fallback_models = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    candidates = [primary_model, *fallback_models]
    seen = set()
    deduplicated = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduplicated.append(candidate)
    return deduplicated


@app.on_event("startup")
async def load_model():
    """Load the LLM model on startup"""
    global llm, tokenizer, model_name, device, max_new_tokens

    # Detect device (Apple MPS or CPU)
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load model name from environment or default
    model_candidates = get_candidate_models()
    model_name = model_candidates[0]
    max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    print(f"Trying LLM models in order: {model_candidates}")

    last_error = None

    for candidate in model_candidates:
        try:
            print(f"Loading LLM model: {candidate}")
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("Loading model (this may take a while)...")
            llm = AutoModelForCausalLM.from_pretrained(
                candidate, low_cpu_mem_usage=True
            ).to(device)
            llm.eval()
            model_name = candidate

            print(f"Model {model_name} loaded successfully!")
            return
        except Exception as e:
            last_error = e
            print(f"Error loading model {candidate}: {e}")
            llm = None
            tokenizer = None

    print("Failed to load all candidate LLM models. Service will stay up but unavailable.")
    if last_error is not None:
        print(f"Last model loading error: {last_error}")


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
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            outputs = llm.generate(inputs, **generation_kwargs)
            generated_tokens = outputs[0][inputs.shape[-1] :]
            response_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
        else:
            inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
            outputs = llm.generate(**inputs, **generation_kwargs)
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            response_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

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
        "max_position_embeddings": (
            llm.config.max_position_embeddings
            if hasattr(llm.config, "max_position_embeddings")
            else "Unknown"
        ),
        "num_parameters": sum(p.numel() for p in llm.parameters() if p.requires_grad),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
