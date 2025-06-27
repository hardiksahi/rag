import sys
import os

## Adding parent (/Users/hardiksahi/Personal/rag/llm_rag_tutorial) of current file to path to enable import
extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(extra_path)
print(extra_path)

from langchain_intro.chatbot import review_chain

if __name__ == "__main__":

    ## RAG invokation (Level of autonomy: LLM call)
    response = review_chain.invoke(
        "What have the patients talked about regarding hospital care?"
    )
    print(response)
