import os
from groq import Groq
from tavily import TavilyClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets when deployed, fall back to environment variables locally."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


# ---------------------------------------------------------------------------
# Cached resource initialisation (Streamlit caches these across reruns)
# ---------------------------------------------------------------------------

def _load_model():
    return SentenceTransformer("sentence-transformers/all-roberta-large-v1")


def _load_clients():
    groq_client = Groq(api_key=_get_secret("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=_get_secret("TAVILY_API_KEY"))
    pc = Pinecone(api_key=_get_secret("PINECONE_API_KEY"))
    index = pc.Index(name=_get_secret("INDEX_NAME"), host=_get_secret("HOST_NAME"))
    return groq_client, tavily, index


# Lazy globals — initialised once per process
_model = None
_groq_client = None
_tavily_client = None
_index = None


def _ensure_loaded():
    global _model, _groq_client, _tavily_client, _index
    if _model is None:
        _model = _load_model()
    if _groq_client is None:
        _groq_client, _tavily_client, _index = _load_clients()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list:
    _ensure_loaded()
    return _model.encode(texts)


def _llm(messages: list[dict], model: str = "openai/gpt-oss-20b") -> str:
    """Call Groq with a messages list (supports system + multi-turn)."""
    _ensure_loaded()
    try:
        completion = _groq_client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Groq API call failed: {e}")


# ---------------------------------------------------------------------------
# Topic guard
# ---------------------------------------------------------------------------

_GUARD_SYSTEM = (
    "You are a strict topic classifier. Reply with exactly one word: YES or NO.\n"
    "Answer YES if the message is about Pakistan Studies, Pakistan history, "
    "the Mughal Empire, the Indian subcontinent, the independence movement, "
    "or related geography/culture.\n"
    "Also answer YES if the message is a follow-up or continuation of a "
    "conversation that was already about those topics (e.g. 'tell me more', "
    "'who was he?', 'explain that', 'what happened next').\n"
    "Answer NO only if the message is clearly unrelated to those topics "
    "and there is no prior relevant conversation context."
)


def _is_relevant_query(user_query: str, chat_history: list[dict]) -> bool:
    messages = [{"role": "system", "content": _GUARD_SYSTEM}]
    # Give the guard the last 4 turns so it can judge follow-ups in context
    for turn in chat_history[-4:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})
    response = _llm(messages)
    return response.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Dynamic rejection
# ---------------------------------------------------------------------------

_REJECTION_SYSTEM = (
    "You are a friendly Pakistan Studies assistant. "
    "The user asked something outside your area of expertise. "
    "Politely decline in 1-2 sentences, mention what topics you *can* help with, "
    "and encourage them to ask a relevant question."
)


def _dynamic_rejection(user_query: str) -> str:
    return _llm([
        {"role": "system", "content": _REJECTION_SYSTEM},
        {"role": "user",   "content": user_query},
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Score thresholds
_BOOK_SCORE_THRESHOLD = 0.60   # prefer book; lower = more inclusive
_WEB_SCORE_THRESHOLD  = 0.60   # minimum Tavily result quality


def get_answer_from_query(
    user_query: str,
    chat_history: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Returns (answer, source) where source is 'Book', 'Internet Search',
    or 'Out of Scope'.

    chat_history: list of {"role": "user"|"assistant", "content": "..."} dicts
                  representing the conversation so far (excluding current query).
    """
    _ensure_loaded()

    if chat_history is None:
        chat_history = []

    # --- 1. Topic guard --------------------------------------------------- #
    if not _is_relevant_query(user_query, chat_history):
        rejection = _dynamic_rejection(user_query)
        return rejection, "Out of Scope"

    # --- 2. Vector retrieval from book ------------------------------------ #
    query_embedding = _embed([user_query])[0]
    query_result = _index.query(vector=query_embedding.tolist(), top_k=5)

    book_context = ""
    used_source = "Internet Search"

    top_score = query_result["matches"][0]["score"] if query_result["matches"] else 0.0

    if top_score >= _BOOK_SCORE_THRESHOLD:
        # Use book passages — join metadata text if available, else vector values
        passages = []
        for match in query_result["matches"]:
            if match["score"] < _BOOK_SCORE_THRESHOLD:
                continue
            text = match.get("metadata", {}).get("text") or " ".join(str(v) for v in match["values"])
            passages.append(text)
        book_context = "\n\n".join(passages)
        used_source = "Book"

    # --- 3. Web fallback (or supplement) ---------------------------------- #
    web_context = ""
    if used_source != "Book":
        try:
            results = _tavily_client.search(user_query)
            relevant = [
                r["content"]
                for r in results.get("results", [])
                if r.get("score", 0) >= _WEB_SCORE_THRESHOLD
            ]
            web_context = "\n\n".join(relevant)
        except Exception:
            web_context = ""

    # --- 4. Build context block ------------------------------------------- #
    context_parts = []
    if book_context:
        context_parts.append(f"[Source: Pakistan Studies Book]\n{book_context}")
    if web_context:
        context_parts.append(f"[Source: Web Search]\n{web_context}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    # --- 5. System prompt ------------------------------------------------- #
    system_prompt = (
        "You are a knowledgeable Pakistan Studies assistant specialising in "
        "the history of Pakistan, the Mughal Empire, the Indian subcontinent, "
        "and the independence movement.\n\n"
        "Rules:\n"
        "- Base your answer strictly on the provided context.\n"
        "- Prefer information from the Pakistan Studies Book over web sources.\n"
        "- Keep answers clear and concise; avoid unnecessary formatting.\n"
        "- Do NOT add <br> tags or HTML.\n"
        "- If the context is insufficient, say so honestly rather than guessing.\n\n"
        f"Context:\n{context}"
    )

    # --- 6. Build message list with history ------------------------------- #
    messages = [{"role": "system", "content": system_prompt}]

    # Include prior conversation turns (cap at last 10 to stay within token limits)
    for turn in chat_history[-10:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user question
    messages.append({"role": "user", "content": user_query})

    # --- 7. Generate answer ----------------------------------------------- #
    answer = _llm(messages)
    return answer, used_source
