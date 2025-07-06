<h1>Langchain based hospital RAG system with Graph DB(Neo4j) integration </h1>

<b>This tutorial is inspired from https://realpython.com/build-llm-rag-chatbot-with-langchain/#step-3-set-up-a-neo4j-graph-database. </b>

<b>The differences are as follows</b>
1. Used latest versions of libraries like Langchain and other related libraries. 
2. Used Ollama for local models hosting instead of OpenAI
3. Model comparison
* QA model: smollm2:1.7b instead of gpt-3.5-turbo-0125
* Agent/ Tool use model: llama3-groq-tool-use:8b instead of gpt-3.5-turbo-1106
* Cypher generation model: tomasonjo/llama3-text2cypher-demo:8b_4bit instead of gpt-3.5-turbo-1106

This ensures that the cost of experimenting with this tutorial is $0

The system runs successfully on LOCAL. This means that Ollama models, FastAPI app and Streamlit app run on LOCAL.
I have tried to run Ollama models separately in 3 Docker containers and FastAPI in a separate container  but with no success (probably due to communication among 3 separate Ollama containers)

Steps to run the chatbot:

0. Rename llm_rag_tutorial/.incomplete_env to llm_rag_tutorial/.env
1. Create AuraDB free instance following steps in https://realpython.com/build-llm-rag-chatbot-with-langchain/#step-3-set-up-a-neo4j-graph-database. Update credentials in  llm_rag_tutorial/.env
2. Populate AuraDB using hospital_neo4j_etl. To do so, execute `docker compose up --build`.
3. Download Ollama to local using https://github.com/ollama/ollama.
4. Download models using `ollama pull smollm2:1.7b,  llama3-groq-tool-use:8b, tomasonjo/llama3-text2cypher-demo:8b_4bit`
5. Open terminal window and cd to `llm_rag_tutorial/chatbot_api/src/` folder. Execute `uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug` to start FastAPI app at port 8000 on local machine
6. Open new terminal window and cd to `llm_rag_tutorial/chatbot_frontend/src/`. Execute `streamlit run main.py` to start streamlit app listening at port 8501.

Note:
Ideally, Ollama models, FastAPI and Streamlit apps should run on Docker. Check docker-compose.yml to see commented lines.


<b>Personal reference</b>
1. Convert pyproject.ml in specific package (chatbot_api, chatbot_frontend, hospital_neo4j_etl) to requirements.txt : `uv export --frozen --no-dev --no-editable --no-hashes  --package hospital-neo4j-etl -o requirements.txt`
2. Add dev dependency to pyproject.ml to active virtual env: `uv add --dev black --active`
3. Mount host volume for use by specific container: `docker run -d --name ollama_container --port <host port>:<docker port> -v $HOME/.ollama/via_docker:/root/.ollama ollama/ollama:0.1.27`



