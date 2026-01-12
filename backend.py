import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(override=True)


@app.post("/run_query")
async def run_query():
    return None


if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn

    load_dotenv(override=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
