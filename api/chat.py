import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Clients are initialised once per cold start (module-level)
_groq_client: Groq | None = None
_tavily_client: TavilyClient | None = None
_pinecone_index = None
_pc: Pinecone | None = None


def _get_clients():
    global _groq_client, _tavily_client, _pinecone_index, _pc
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    if _pc is None:
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if _pinecone_index is None:
        _pinecone_index = _pc.Index(
            name=os.environ["INDEX_NAME"],
            host=os.environ["HOST_NAME"],
        )
    return _groq_client, _tavily_client, _pc, _pinecone_index


def _generate_embeddings(pc: Pinecone, texts: list[str]) -> list[list[float]]:
    """Use Pinecone-hosted inference instead of local sentence-transformers."""
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={"input_type": "query"},
    )
    return [item["values"] for item in result]


def _llm_call(groq_client: Groq, prompt: str) -> str:
    completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    return completion.choices[0].message.content


def _is_pakistan_history_query(groq_client: Groq, query: str) -> bool:
    prompt = (
        "Analyze the following Question and give one word answer of YES if the question "
        "is about Pakistan history, the Mughal empire, events in the Indian subcontinent "
        "after the Mughal empire, or the formation of Pakistan. Answer with NO otherwise.\n"
        f"Question: {query}"
    )
    return _llm_call(groq_client, prompt).strip().upper() == "YES"


# ---------- request / response models ----------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    source: str


# ---------- endpoint ----------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    groq_client, tavily_client, pc, index = _get_clients()

    if not req.query.strip():
        return ChatResponse(answer="Please enter a question.", source="Error")

    if not _is_pakistan_history_query(groq_client, req.query):
        return ChatResponse(
            answer="This chatbot only answers questions about Pakistan Studies and History.",
            source="Error",
        )

    query_vector = _generate_embeddings(pc, [req.query])[0]
    query_result = index.query(vector=query_vector, top_k=3)

    if query_result["matches"] and query_result["matches"][0]["score"] > 0.65:
        context = "\n".join(
            [" ".join(match["values"]) for match in query_result["matches"]]
        )
        source = "Book"
    else:
        search_result = tavily_client.search(req.query)
        relevant = [
            r["content"]
            for r in search_result["results"]
            if r.get("score", 0) > 0.65
        ]
        context = "\n\n".join(relevant)
        source = "Internet Search"

    answer_prompt = (
        f"Context:\n{context}\n\n"
        f"User Query: {req.query}\n\n"
        "Restrictions: Don't over-format the result. Don't add <br> tokens. "
        "Don't add info other than the context.\n\n"
        "Answer:"
    )
    answer = _llm_call(groq_client, answer_prompt)
    return ChatResponse(answer=answer, source=source)
