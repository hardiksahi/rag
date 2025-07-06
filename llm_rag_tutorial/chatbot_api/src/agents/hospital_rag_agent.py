import os
import sys
import dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, Tool, AgentExecutor

from langchain import hub


##extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) ##/Users/hardiksahi/Personal/rag/llm_rag_tutorial/
extra_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  ##/Users/hardiksahi/Personal/rag/llm_rag_tutorial/chatbot_api/src
sys.path.append(extra_path)

from chains.hospital_review_chain import review_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

# from chains.hospital_cypher_chain import hospital_cypher_chain

dotenv.load_dotenv()

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")
BASE_OLLAMA_AGENT_URL = os.getenv("BASE_OLLAMA_AGENT_URL")

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tool_list = [
    Tool(
        name="Experiences",
        func=lambda q: review_chain.invoke({"input": q})["answer"],
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=lambda q: hospital_cypher_chain.invoke({"query": q})["result"],
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]

agent_model = ChatOllama(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
    base_url=BASE_OLLAMA_AGENT_URL,
    verbose=True,
)
hospital_rag_agent = create_tool_calling_agent(
    llm=agent_model, prompt=agent_prompt, tools=tool_list
)
hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tool_list,
    return_intermediate_steps=True,
    verbose=True,
)

if __name__ == "__main__":
    ## Invoking review chain tool
    # response = hospital_rag_agent_executor.invoke(
    #     {
    #         "input": "What have patients said about how doctors and nurses communicate with them?"
    #     }
    # )

    ## Invoking wait time related tools
    # response = hospital_rag_agent_executor.invoke(
    #     {"input": "What is the wait time at Wallace-Hamilton?"}
    # )

    ## Invoking Graph
    # response = hospital_rag_agent_executor.invoke(
    #     {
    #         "input": "Query the graph database to find which physician has treated the most number of distinct visits covered by Cigna?"
    #     }
    # )

    # response = hospital_rag_agent_executor.invoke(
    #     {"input": "Show me reviews written by patient 7674."}
    # )

    ## Invoking Graph by specifically asking agent to use graph database
    response = hospital_rag_agent_executor.invoke(
        {
            "input": "Query the graph database to show me the reviews written by patient 7674. Use text property from review node"
        }
    )

    print(response.keys())
    print("*************")
    print(f"Input: {response.get('input')}")
    print("*************")
    print(f"Output: {response.get('output')}")
    print("*************")
    print(f"intermediate_steps: {response.get('intermediate_steps')}")
