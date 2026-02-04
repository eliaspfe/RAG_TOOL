import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
import shutil

from ragpipeline import RagPipeline

pipeline = RagPipeline()
UPLOAD_DIR = "./uploads"

app = FastAPI()


# service:
# - embedding model
# - frontend
# backend: inklusive RAGPIPLINE Klasse

# pipline = RAGPIPLINE()

# funktionen der Klasse
# pipeline.ducklake(Dateipfad) Noahs Teil -> 3 Layers, Daten werden aus den PDFs extrahiert und in DuckDB gespeichert
# pipline.embed_chunks_and_save_to_duckdb() Felix Teil -> Chunks laufen durch das Embedding Model und werden in DuckDB gespeichert
# pipline.build_prompt_with_context(user_query) -> User Prompt wird Embedded, ähnlichkeitssuche in der DuckDB, Kontext wird zurückgegeben (string)


SYS_PROMPT = "You are a helpful assistant."

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(override=True)

# initialize in-memory saver for message history
checkpointer = InMemorySaver()


# initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    # ... (other params)
)
agent = create_agent(model, system_prompt=SYS_PROMPT, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}


class LLMRequest(BaseModel):
    query: str


def get_latest_ai_message(messages) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


@app.post("/build_index")
def build_index():
    try:
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            pipeline.pdf_chunk_and_store(file_path)

        pipeline.load_and_embed_chunks()
        return {"status": "Index built successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test Ragpipeline Object
@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "PDF saved", "filename": file.filename}


@app.post("/run_query")
def run_query(request: LLMRequest) -> dict:
    # 1. Retreive context from pdf document
    # context = retrieve_context_from_pdf(request.query)
    # 2. Build Prompt LLM with context and user query

    # 3. Invoke LLM with the prompt
    final_prompt = pipeline.build_prompt(request.query, top_k=5)
    print(pipeline.similarity_search(request.query, top_k=5))
    print(final_prompt)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": final_prompt}]},
        config=config,
    )

    latest_ai = get_latest_ai_message(response["messages"])

    return {"content": latest_ai.content}


if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn

    load_dotenv(override=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
