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
