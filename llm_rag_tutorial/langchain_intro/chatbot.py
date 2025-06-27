import dotenv
from langchain_ollama import ChatOllama
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

# from create_retriever import REVIEWS_CHROMA_PATH, embedding
from constants import REVIEWS_CHROMA_PATH, EMBEDDER_PARAMS, TOP_K
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain import hub
from langchain_intro.tools import get_current_wait_time


dotenv.load_dotenv()

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

## Step 1. Provide system prompt
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template_str)
)

## Step 2: Provide human prompt
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)

## Step 3: Compose message list
message_list = [review_system_prompt, review_human_prompt]

## Step 4: Final review prompt template
review_prompt_template = ChatPromptTemplate.from_messages(messages=message_list)

## Step 5: Initialize chat model using ollama
chat_model = ChatOllama(model="llama3.2", temperature=0)

## Step 7: Load chroma db
embedder = HuggingFaceEmbeddings(**EMBEDDER_PARAMS)
reviews_vector_db = Chroma(
    embedding_function=embedder, persist_directory=REVIEWS_CHROMA_PATH
)

reviews_retreiver = reviews_vector_db.as_retriever(k=TOP_K)

## Step 6: Review chain
review_chain = (
    {"context": reviews_retreiver, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

review_tool = Tool(
    name="Reviews",
    func=review_chain.invoke,
    description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
)
wait_tool = Tool(
    name="Waits",
    func=get_current_wait_time,
    description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
)
tool_list = [review_tool, wait_tool]

agent_chat_model = ChatOllama(
    model="llama3-groq-tool-use:8b", temperature=0
)  ## llama3.2:latest

prompt = hub.pull("hwchase17/openai-functions-agent")
hospital_agent = create_tool_calling_agent(
    llm=agent_chat_model, prompt=prompt, tools=tool_list
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent, tools=tool_list, return_intermediate_steps=True, verbose=True
)


if __name__ == "__main__":
    ## Way 1: Non modular way of creating prompts composed of System message + Human message
    # messages = [
    #     SystemMessage(
    #         content="""You are an assistant knowledgable about healthcare. Only answer healthcare related questions"""
    #     ),
    #     HumanMessage(content="What is Medicaid managed care?"),
    # ]
    # ai_response = chat_model.invoke(messages)
    # print(f"Answer: {ai_response.content}")

    #############################################################

    ## Way 2: Modular way using prompt templates (predefined recipes for generating prompts for language models)
    ## Modular because we have a template where we can change context and question dynamically
    ## This is what I did in my initial rag notebook.
    ## Directly using ChatPromptTemplate => message from human => not sensible

    # review_template_str = """Your job is to use patient
    # reviews to answer questions about their experience at a hospital.
    # Use the following context to answer questions. Be as detailed
    # as possible, but don't make up any information that's not
    # from the context. If you don't know an answer, say you don't know.

    # {context}

    # {question}
    # """
    # review_template = ChatPromptTemplate.from_template(
    #     template=review_template_str
    # )  ## By default it is HumanMessage
    # context = "I had a great stay!"
    # question = "Did anyone have a positive experience?"

    # # print(review_template.format(context=context, question=question))

    # chainn = review_template | chat_model

    # ai_response = chainn.invoke({"question": question, "context": context})

    # print(f"Answer: {ai_response.content}")

    #############################################################

    ## Way 3: Modular with appropriate prompt templates (system, human etc)
    ## Main code outside main

    # context = "I had a great stay!"
    # question = "What is the patient talking about?"
    # print(review_prompt_template.format(context=context, question=question))
    # print("***************")
    # ai_response_str = review_chain.invoke({"context": context, "question": question})
    # print(f"Answer: {ai_response_str}")

    #############################################################

    ## Utility 1: Similarity search from ChromaDB.
    # question = """Has anyone complained about
    #        communication with the hospital staff?"""
    # relevant_docs = reviews_vector_db.similarity_search(question, k=10)

    # for i in range(len(relevant_docs)):
    #     print(relevant_docs[i].page_content)
    #     print("************************")
    # print(
    #     review_prompt_template.format(
    #         context=reviews_retreiver,
    #         question="Have there been any complains during hospital visits?",
    #     )
    # )

    ## Utility 2 RAG invokation (LLM call)
    # response = review_chain.invoke(
    #     "What have the patients talked about regarding hospital care?"
    # )
    # print(response)

    ## Utility 3: Router (Agent with toolcalling)
    # hospital_agent_executor.invoke(
    #     {"input": "What have patients said about their comfort at the hospital?"}
    # )
    pass
