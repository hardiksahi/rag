import dotenv
from langchain_neo4j import (
    Neo4jVector,
)  ## https://python.langchain.com/api_reference/neo4j/vectorstores/langchain_neo4j.vectorstores.neo4j_vector.Neo4jVector.html#langchain_neo4j.vectorstores.neo4j_vector.Neo4jVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

## Following are replacement of RetrievalQA
## https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html
from langchain.chains.retrieval import create_retrieval_chain

## https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
import sys

# extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) ##/Users/hardiksahi/Personal/rag/llm_rag_tutorial/
extra_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  ## /Users/hardiksahi/Personal/rag/llm_rag_tutorial//chatbot_api/src
sys.path.append(extra_path)
from utils.constants import EMBEDDER_PARAMS, REVIEW_TOP_K

dotenv.load_dotenv()

## The idea is to use the graph created in Neo4j Aura instance and do the following:
# 1. Create embeddings using open/ closed embedders like sentence transformer and store it to graph db (using Neo4jVector)
# 2. Create review prompt
# 3. Use neo4j as retreiver (in intro, it was chromadb that stored and retreived embeddings)
# 4. Create review chain that uses embeddings to answer natural language questions.

## Step 1: Create embeddings for Review node properties
embedder = HuggingFaceEmbeddings(**EMBEDDER_PARAMS)  ## Sentence transformer embedding
vectorestore = Neo4jVector.from_existing_graph(
    embedding=embedder,
    index_name="reviews",
    node_label="Review",
    text_node_properties=["physician_name", "patient_name", "text", "hospital_name"],
    embedding_node_property="embedding",
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

## Step 2: Create review prompt using template
review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template_str)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["input"], template="{input}"
    )  ## Mandatory to use {input} as placeholder due to hardcoded implemntation in create_retrieval_chain
)

message_list = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate.from_messages(messages=message_list)

## Step 3: Set up chatmodel and use neo4j_vector as retreiver
chat_model = ChatOllama(
    model=os.getenv("HOSPITAL_QA_MODEL"),
    temperature=0,
    base_url=os.getenv("BASE_OLLAMA_QA_URL"),
    verbose=True,
)
combine_docs_chain = create_stuff_documents_chain(
    llm=chat_model, prompt=review_prompt_template, document_variable_name="context"
)
review_chain = create_retrieval_chain(
    retriever=vectorestore.as_retriever(fetch_k=REVIEW_TOP_K),
    combine_docs_chain=combine_docs_chain,
)

## Create chain on your own. Flexibility in deciding placeholder names
# review_chain = (
#     {
#         "context": vectorestore.as_retriever(fetch_k=TOP_K),
#         "question": RunnablePassthrough(),
#     }
#     | review_prompt_template
#     | chat_model
#     | StrOutputParser()
# )
# res = review_chain.invoke(
#         "What have the patients talked about regarding hospital care?"
#     )

if __name__ == "__main__":
    res = review_chain.invoke(
        {
            "input": "What have patients said about how doctors and nurses communicate with them?"
        }
    )  ## Has to be dict with key as input due to hardcoding
    print(res["answer"])
