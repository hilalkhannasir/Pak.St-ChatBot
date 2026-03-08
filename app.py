import streamlit as st
from data_retrieval import get_answer_from_query

st.set_page_config(page_title="Pakistan History Chatbot", page_icon="📚")

st.title("Study Pakistan History with AI")
st.caption("Trained on O Level Pakistan Studies History Book with additional Web Search Capability")

if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

user_query = st.chat_input("Ask a question about Pakistan history...")

if user_query:

    st.session_state.messages.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.write("AI is thinking...")

        try:
            answer, source = get_answer_from_query(user_query)
            thinking.empty()
            st.write(answer)
            st.caption(f"Source: {source}")

            st.session_state.messages.append(("assistant", answer))

        except Exception as e:
            thinking.empty()
            st.write(f"Error: {e}")