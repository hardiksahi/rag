import sys
import os

## Adding parent (/Users/hardiksahi/Personal/rag/llm_rag_tutorial) of current file to path to enable import
extra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(extra_path)
from langchain_intro.chatbot import hospital_agent_executor

if __name__ == "__main__":
    ## Level of autonomy: Router
    hospital_agent_executor.invoke(
        {
            # "input": "What kind of reviews have patients made for communication with staff?"
            "input": "What is the wait time for hospital ABC?"
        }
    )
