import streamlit as st
from data_retrieval import get_answer_from_query

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistan Studies Chatbot",
    page_icon="🇵🇰",
    layout="centered",
)

st.title("🇵🇰 Pakistan Studies AI Assistant")
st.caption("Powered by an O-Levels Pakistan Studies textbook · Ask about history, geography, and culture")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "source": str|None}
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("source"):
            source = msg["source"]
            if source == "Book":
                st.caption("📚 Source: Pakistan Studies Book")
            elif source == "Internet Search":
                st.caption("🌐 Source: Web Search")
            # Out of Scope gets no source caption — the message itself explains it

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about Pakistan history...")

if user_query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query, "source": None})
    with st.chat_message("user"):
        st.write(user_query)

    # Build history to pass (all turns except the one just added)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude current user message
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("⏳ Thinking...")

        try:
            answer, source = get_answer_from_query(user_query, chat_history=history)
            placeholder.empty()
            st.write(answer)

            if source == "Book":
                st.caption("📚 Source: Pakistan Studies Book")
            elif source == "Internet Search":
                st.caption("🌐 Source: Web Search")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "source": source}
            )

        except Exception as e:
            placeholder.empty()
            error_msg = f"Something went wrong: {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "source": None}
            )

# ---------------------------------------------------------------------------
# Sidebar — clear history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "**Topics I can help with:**\n"
        "- Mughal Empire & decline\n"
        "- Indian subcontinent history\n"
        "- Pakistan independence movement\n"
        "- Pakistan Studies O-Level syllabus\n"
        "- Pakistani geography & culture"
    )
