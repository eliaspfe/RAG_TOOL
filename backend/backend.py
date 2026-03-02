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
from dataLake import DataLake
import requests

load_dotenv(override=True)

data_lake = DataLake()
pipeline = RagPipeline()
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
sources = []

app = FastAPI()
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://llm-service:8002")


SYS_PROMPT = "You are a helpful assistant."

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_TYPE = os.getenv("LLM_TYPE", "api").strip().lower()  # 'local' oder 'api'

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
            data_lake.process_document(file_path=file_path, doc_name=f)
            print("Processed:", f)

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

    data_lake.process_document(file_path=file_path, doc_name=file.filename)
    print("Processed:", file.filename)
    return {
        "status": "PDF saved and Index built successfully",
        "filename": file.filename,
    }


@app.post("/run_query")
def run_query(request: LLMRequest) -> dict:
    # 1. Retreive context from pdf document
    # context = retrieve_context_from_pdf(request.query)
    # 2. Build Prompt LLM with context and user query
    final_prompt = pipeline.build_prompt(request.query, top_k=5)

    # 3. Invoke LLM with the prompt
    if LLM_TYPE == "api":
        print("Quellen:")
        anfrage = pipeline.similarity_search(request.query, top_k=5)
        sources = [row["doc_name"] for row in anfrage]
        print("\n".join(sources))
        print(final_prompt)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": final_prompt}]},
            config=config,
        )

        latest_ai = get_latest_ai_message(response["messages"])

        return {"content": latest_ai.content}
    if LLM_TYPE == "local":
        try:
            response = requests.post(
                f"{LOCAL_LLM_URL}/query",
                json={"prompt": final_prompt},
                timeout=180,
            )
            response.raise_for_status()
            response_data = response.json()
            return {"content": response_data.get("response", "")}
        except requests.ConnectionError as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Local LLM service not reachable at {LOCAL_LLM_URL}. "
                    "If backend runs in Docker, use http://llm-service:8002; "
                    "if backend runs locally, use http://localhost:8002. "
                    f"Original error: {e}"
                ),
            )
        except requests.Timeout:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Local LLM request timed out at {LOCAL_LLM_URL}. "
                    "The model may still be loading."
                ),
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources_from_last_query")
def sources_from_last_query():
    if not sources:
        raise HTTPException(status_code=404, detail="No sources found from last query")
    unique_sources = list(dict.fromkeys(sources))
    return {"sources": unique_sources}


@app.get("/list_pdfs")
def list_pdfs():
    files = []
    for f in os.listdir(UPLOAD_DIR):
        if f.lower().endswith(".pdf"):
            files.append(f)
    return {"files": files}


@app.post("/delete_index")
def delete_index():
    try:
        data_lake.remove_all_data()
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        return {"status": "Index gelöscht"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn

    load_dotenv(override=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
