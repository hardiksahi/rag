from fastapi import FastAPI
import os
import sys
import httpx
import asyncio

extra_path = os.path.abspath(
    os.path.dirname(__file__)
)  ##/Users/hardiksahi/Personal/rag/llm_rag_tutorial/chatbot_api/src
sys.path.append(extra_path)

from agents.hospital_rag_agent import hospital_rag_agent_executor
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput

app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/async-hospital-rag-agent")
async def async_query_hospital_rag(query: HospitalQueryInput) -> HospitalQueryOutput:
    query_response = await hospital_rag_agent_executor.ainvoke({"input": query.text})
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]
    return query_response


## Shows how to convert sync -> async call using asyncio if there was no ainvoke method
@app.post("/sync-to-async-hospital-rag-agent")
async def sync_to_async_query_hospital_rag(
    query: HospitalQueryInput,
) -> HospitalQueryOutput:

    loop = asyncio.get_event_loop()
    query_response = await loop.run_in_executor(
        None, hospital_rag_agent_executor.invoke, {"input": query.text}
    )
    # query_response = hospital_rag_agent_executor.invoke({"input": query.text})
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]
    return query_response


## DO NOT KNOW WHAT SYNC MEANS!
@app.post("/sync-hospital-rag-agent")
def sync_query_hospital_rag(
    query: HospitalQueryInput,
) -> HospitalQueryOutput:

    query_response = hospital_rag_agent_executor.invoke({"input": query.text})
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]
    return query_response


# How many visits were made to Wallace-Hamilton hospital in 2023?
# Query the graph database to show me the reviews written by patient 7674. Use text property from review node
