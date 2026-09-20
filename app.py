from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict
from urllib.parse import quote_plus

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda left, right: left + right]


CRICKET_DB = os.getenv("CRICKET_DB", "data/chroma")


def cricket_store() -> Chroma | None:
    if not os.path.isdir(CRICKET_DB):
        return None
    return Chroma(
        collection_name="cricket_knowledge",
        persist_directory=CRICKET_DB,
        embedding_function=OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ),
    )


@tool
def cricket_knowledge_search(query: str) -> str:
    """Search the local cricket knowledge base for rules, history, profiles, and match facts."""
    store = cricket_store()
    if store is None:
        return "Cricket knowledge base is not indexed. Run: python ingest_cricket.py"
    documents = store.similarity_search(query, k=5)
    if not documents:
        return "No relevant cricket documents were found."
    return "\n\n---\n\n".join(document.page_content for document in documents)


@tool
def web_search(query: str) -> str:
    """Search the public web for current information and return concise results."""
    try:
        response = httpx.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; LocalAgent/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Search unavailable: {exc}"

    from html.parser import HTMLParser

    class ResultParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.results: list[str] = []
            self.current: list[str] = []
            self.active = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            classes = dict(attrs).get("class", "") or ""
            self.active = "result__title" in classes or "result__snippet" in classes
            if self.active:
                self.current = []

        def handle_endtag(self, tag: str) -> None:
            if self.active:
                text = " ".join("".join(self.current).split())
                if text:
                    self.results.append(text)
                self.active = False
                self.current = []

        def handle_data(self, data: str) -> None:
            if self.active:
                self.current.append(data)

    parser = ResultParser()
    parser.feed(response.text)
    return "\n".join(parser.results[:8]) or "No search results found."


def build_graph():
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        temperature=0.2,
    )
    model_with_tools = model.bind_tools([cricket_knowledge_search, web_search])

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def run_tools(state: AgentState) -> dict[str, list[BaseMessage]]:
        last = state["messages"][-1]
        messages: list[ToolMessage] = []
        for call in last.tool_calls:
            if call["name"] == "web_search":
                result = web_search.invoke(call["args"])
            elif call["name"] == "cricket_knowledge_search":
                result = cricket_knowledge_search.invoke(call["args"])
            else:
                continue
            messages.append(ToolMessage(content=result, name=call["name"], tool_call_id=call["id"]))
        return {"messages": messages}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    return graph.compile()


app = FastAPI(title="SenguptaChat")
GRAPH = None


def event(kind: str, **payload: object) -> str:
    return f"data: {json.dumps({'type': kind, **payload})}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/graph")
def graph_page() -> FileResponse:
    return FileResponse("static/graph.html")


@app.get("/api/graph")
def graph_definition() -> dict[str, str]:
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return {"mermaid": GRAPH.get_graph().draw_mermaid()}


@app.get("/static/{path:path}")
def static_files(path: str) -> FileResponse:
    return FileResponse(f"static/{path}")


@app.post("/api/chat")
def chat(payload: dict[str, object]) -> StreamingResponse:
    user_message = str(payload.get("message", "")).strip()
    history = payload.get("history", [])

    def stream():
        global GRAPH
        if not user_message:
            yield event("error", message="Write a message first.")
            return
        if GRAPH is None:
            try:
                GRAPH = build_graph()
            except Exception as exc:
                yield event("error", message=f"The local agent is not ready. Start Ollama and pull a model. Details: {exc}")
                return

        messages: list[BaseMessage] = [SystemMessage(content=(
            "You are an expert cricket analysis assistant. Use cricket_knowledge_search "
            "for historical facts, rules, player profiles, and match records. Use web_search "
            "for current scores, news, injuries, and recent matches. Never invent statistics; "
            "separate sourced facts from your analysis."
        ))]
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                messages.append(HumanMessage(content=str(item.get("content", ""))))
            elif item.get("role") == "assistant":
                messages.append(AIMessage(content=str(item.get("content", ""))))
        messages.append(HumanMessage(content=user_message))
        yield event("status", label="Thinking", detail="Planning the next step")

        try:
            for update in GRAPH.stream({"messages": messages}, stream_mode="updates"):
                for node, data in update.items():
                    if node == "tools":
                        tool_names = {getattr(message, "name", "") for message in data["messages"]}
                        if "cricket_knowledge_search" in tool_names:
                            yield event("tool", label="Cricket knowledge", detail="Searching the local cricket index")
                        else:
                            yield event("tool", label="Web search", detail="Gathering fresh context")
                    elif node == "agent":
                        latest = data["messages"][-1]
                        if isinstance(latest, AIMessage) and latest.tool_calls:
                            yield event("status", label="Choosing tools", detail="A search will improve the answer")
                        elif isinstance(latest, AIMessage):
                            yield event("message", content=latest.content)
            yield event("done")
        except Exception as exc:
            if "Connection refused" in str(exc) or "Failed to connect" in str(exc):
                yield event(
                    "error",
                    message=(
                        "Ollama is not running at http://127.0.0.1:11434. "
                        "Start Ollama, pull the model, and try again."
                    ),
                )
            else:
                yield event("error", message=f"Agent error: {exc}")

    return StreamingResponse(stream(), media_type="text/event-stream")