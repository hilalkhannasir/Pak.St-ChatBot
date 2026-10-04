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

_groq_client: Groq | None = None
_tavily_client: TavilyClient | None = None
_pinecone_index = None
_pc: Pinecone | None = None

SYSTEM_PROMPT = """You are an expert tutor on Pakistan Studies and History, specialised in the O-Level Pakistan Studies syllabus.

Your knowledge covers:
- The Mughal Empire and its decline
- The role of the East India Company and British colonial rule in the Indian Subcontinent
- The struggle for independence: key figures (Sir Syed Ahmad Khan, Allama Iqbal, Quaid-e-Azam Muhammad Ali Jinnah), movements, and events
- The Partition of 1947 and the formation of Pakistan
- Post-independence history of Pakistan

Guidelines for your answers:
- Answer strictly from the provided context. Do not introduce facts not present in the context.
- Be clear, accurate, and concise. Avoid unnecessary padding or filler.
- Write in plain prose. Do not use markdown headers, bullet points, or HTML tags like <br>.
- If the context is insufficient to fully answer the question, say so honestly rather than guessing.
- Maintain awareness of the conversation history to give coherent follow-up answers.
- Address the student directly and use an encouraging, educational tone."""


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
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={"input_type": "query"},
    )
    return [item["values"] for item in result]


def _is_pakistan_history_query(groq_client: Groq, query: str) -> bool:
    """Single-purpose classifier — kept separate so it doesn't pollute chat history."""
    completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query classifier. Reply with exactly one word: "
                    "YES if the question is about Pakistan history, the Mughal empire, "
                    "events in the Indian subcontinent after the Mughal empire, or the "
                    "formation of Pakistan. Reply NO otherwise."
                ),
            },
            {"role": "user", "content": query},
        ],
        model="openai/gpt-oss-20b",
        max_tokens=5,
    )
    return completion.choices[0].message.content.strip().upper() == "YES"


def _build_messages(history: list[dict], context: str, query: str) -> list[dict]:
    """
    Construct the messages array for the LLM:
      system  → role + rules + retrieved context for this turn
      history → prior turns (trimmed to last 10 to stay within token limits)
      user    → current question
    """
    system_with_context = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Retrieved Context for this question ---\n{context}\n"
        f"-------------------------------------------"
    )

    # Keep only the last 10 messages (5 turns) to avoid token overflow
    recent_history = history[-10:] if len(history) > 10 else history

    messages = [{"role": "system", "content": system_with_context}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": query})
    return messages


# ---------- request / response models ----------

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[Message] = []


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
            answer="I can only answer questions about Pakistan Studies and History. Please ask something related to the subject.",
            source="Error",
        )

    # Retrieve context
    query_vector = _generate_embeddings(pc, [req.query])[0]
    query_result = index.query(vector=query_vector, top_k=3, include_metadata=True)

    if query_result["matches"] and query_result["matches"][0]["score"] > 0.65:
        context = "\n".join(
            match["metadata"].get("content", "") for match in query_result["matches"]
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

    # Build messages with history and call LLM
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    messages = _build_messages(history_dicts, context, req.query)

    completion = groq_client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-20b",
    )
    answer = completion.choices[0].message.content

    return ChatResponse(answer=answer, source=source)
