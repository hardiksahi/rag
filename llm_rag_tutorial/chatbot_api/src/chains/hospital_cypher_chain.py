## Accept user natural language query
## COnvert to Cypher
## Execute query against Neo4J database and retreive results.
## Use retreived results to answer the question
## ALl this possible with GraphCypherQAChain (https://python.langchain.com/api_reference/neo4j/chains/langchain_neo4j.chains.graph_qa.cypher.GraphCypherQAChain.html#)
## We will experiment with prompt tuning to generate cypher queries
## Alternative: Use LLM finetuned on Cypher query dataset

import dotenv
import os
import sys
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_neo4j.chains.graph_qa.cypher import construct_schema
from langchain_ollama import ChatOllama
from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
)

# extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) ## ## /Users/hardiksahi/Personal/rag/llm_rag_tutorial/
extra_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  ## /Users/hardiksahi/Personal/rag/llm_rag_tutorial//chatbot_api/src
sys.path.append(extra_path)
from utils.constants import CYPHER_TOP_K

dotenv.load_dotenv()

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
HOSPITAL_CYPHER_MODEL = os.getenv("HOSPITAL_CYPHER_MODEL")
BASE_OLLAMA_CYPHER_URL = os.getenv("BASE_OLLAMA_CYPHER_URL")
BASE_OLLAMA_QA_URL = os.getenv("BASE_OLLAMA_QA_URL")

print(
    f"HOSPITAL_QA_MODEL and BASE_OLLAMA_QA_URL: {HOSPITAL_QA_MODEL}, {BASE_OLLAMA_QA_URL}"
)
print(
    f"HOSPITAL_CYPHER_MODEL and BASE_OLLAMA_CYPHER_URL: {HOSPITAL_CYPHER_MODEL}, {BASE_OLLAMA_CYPHER_URL}"
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    sanitize=False,  ## To avoid getting embedding properties, set sanitize as True
)
graph.refresh_schema()

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH v as visit, c.billing_amount as billing_amount)
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Examples:
# Who is the oldest patient and how old are they?
MATCH (p:Patient)
RETURN p.name AS oldest_patient,
       duration.between(date(p.dob), date()).years AS age
ORDER BY age DESC
LIMIT 1

# Which physician has billed the least to Cigna
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[t:TREATS]-(phy:Physician)
WHERE p.name = 'Cigna'
RETURN phy.name AS physician_name, SUM(c.billing_amount) AS total_billed
ORDER BY total_billed
LIMIT 1

# Which state had the largest percent increase in Cigna visits
# from 2022 to 2023?
MATCH (h:Hospital)<-[:AT]-(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Cigna' AND v.admission_date >= '2022-01-01' AND
v.admission_date < '2024-01-01'
WITH h.state_name AS state, COUNT(v) AS visit_count,
     SUM(CASE WHEN v.admission_date >= '2022-01-01' AND
     v.admission_date < '2023-01-01' THEN 1 ELSE 0 END) AS count_2022,
     SUM(CASE WHEN v.admission_date >= '2023-01-01' AND
     v.admission_date < '2024-01-01' THEN 1 ELSE 0 END) AS count_2023
WITH state, visit_count, count_2022, count_2023,
     (toFloat(count_2023) - toFloat(count_2022)) / toFloat(count_2022) * 100
     AS percent_increase
RETURN state, percent_increase
ORDER BY percent_increase DESC
LIMIT 1

# How many non-emergency patients in North Carolina have written reviews?
match (r:Review)<-[:WRITES]-(v:Visit)-[:AT]->(h:Hospital)
where h.state_name = 'NC' and v.admission_type <> 'Emergency'
return count(*)

String category values:
Test results are one of: 'Inconclusive', 'Normal', 'Abnormal'
Visit statuses are one of: 'OPEN', 'DISCHARGED'
Admission Types are one of: 'Elective', 'Emergency', 'Urgent'
Payer names are one of: 'Cigna', 'Blue Cross', 'UnitedHealthcare', 'Medicare',
'Aetna'

A visit is considered open if its status is 'OPEN' and the discharge date is
missing.
Use abbreviations when
filtering on hospital states (e.g. "Texas" is "TX",
"Colorado" is "CO", "North Carolina" is "NC",
"Florida" is "FL", "Georgia" is "GA, etc.)

Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
Never return embedding properties in your queries. You must never include the
statement "GROUP BY" in your query. Make sure to alias all statements that
follow as with statement (e.g. WITH v as visit, c.billing_amount as
billing_amount)
If you need to divide numbers, make sure to filter the denominator to be non
zero.

The question is:
{question}
"""
# cypher_generation_template = """
#     Generate Cypher statement to query a graph database.
#     Use only the provided relationship types and properties in the schema.
#     Schema: {schema}
#     Question: {question}
#     Cypher output: "
# """

cypher_prompt = PromptTemplate(
    template=cypher_generation_template, input_variables=["schema", "question"]
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as hospital names,
beware  of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single hospital name,
not multiple hospitals. Make sure you return any list of names in
a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Make sure to show all the relevant query results
if you're asked.

Helpful Answer:
"""
qa_prompt = PromptTemplate(
    template=qa_generation_template, input_variables=["context", "question"]
)

cypher_model = ChatOllama(
    model=HOSPITAL_CYPHER_MODEL,
    temperature=0,
    base_url=BASE_OLLAMA_CYPHER_URL,
    verbose=True,
)
qa_model = ChatOllama(
    model=HOSPITAL_QA_MODEL, temperature=0, base_url=BASE_OLLAMA_QA_URL, verbose=True
)
hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=cypher_model,
    qa_llm=qa_model,
    graph=graph,
    qa_prompt=qa_prompt,
    cypher_prompt=cypher_prompt,
    validate_cypher=True,
    verbose=True,
    top_k=CYPHER_TOP_K,
    return_intermediate_steps=False,
    allow_dangerous_requests=True,
)

if __name__ == "__main__":
    #     query = """MATCH (n)
    # WHERE n.embedding IS NOT NULL
    # RETURN DISTINCT "node" as entity, n.embedding AS embedding LIMIT 25
    # UNION ALL
    # MATCH ()-[r]-()
    # WHERE r.embedding IS NOT NULL
    # RETURN DISTINCT "relationship" AS entity, r.embedding AS embedding LIMIT 25;"""
    #     res = graph.query(query=query)
    #     print(res)
    # print(graph.get_structured_schema)
    # print("==================")
    # print(graph._enhanced_schema)
    # print("**************************")
    # graph_schema = construct_schema(
    #     graph.get_structured_schema,
    #     [],
    #     [],
    #     graph._enhanced_schema,
    # )
    # print(graph_schema)

    result = hospital_cypher_chain.invoke(
        {
            "query": "How many total visits in 2023?"
        }  ##"How much was billed for patient 789’s stay?"
    )
    print(result.get("result"))
