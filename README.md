# SenCrickInfo

A small open-source, local agentic chat loop built with LangGraph, Ollama, FastAPI, and a no-key DuckDuckGo HTML search tool.

## Run it

1. Install [Ollama](https://ollama.com), start it, and pull a model:

   ```sh
   ollama pull llama3.1:8b
   ```

2. Create an environment and install dependencies:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Start the app:

   ```sh
   uvicorn app:app --reload
   ```

Open http://127.0.0.1:8000. Set `OLLAMA_MODEL` to use another model.

## Cricket RAG

Pull the local embedding model and index the Markdown files under `data/cricket`:

```sh
ollama pull nomic-embed-text
python ingest_cricket.py
```

The agent uses the cricket index for rules, history, player profiles, and match records. It uses web search for current scores, news, injuries, and recent matches.