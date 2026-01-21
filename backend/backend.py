import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

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


@app.post("/run_query")
def run_query(request: LLMRequest) -> dict:
    # 1. Retreive context from pdf document
    # context = retrieve_context_from_pdf(request.query)
    # 2. Build Prompt LLM with context and user query

    # 3. Invoke LLM with the prompt
    response = agent.invoke(
        {"messages": [{"role": "user", "content": request.query}]},
        config=config,
    )

    latest_ai = get_latest_ai_message(response["messages"])

    return {"content": latest_ai.content}


if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn

    load_dotenv(override=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
