import streamlit as st
from data_retrieval import get_answer_from_query

st.set_page_config(page_title="Pakistan History Chatbot", page_icon="📚")

st.markdown("""
<style>

[data-testid="stChatInput"] {
    border: 2px solid #1f6fff !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea {
    border: 2px solid #1f6fff !important;
    border-radius: 8px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: 2px solid #1f6fff !important;
    box-shadow: 0 0 0 2px rgba(31,111,255,0.3) !important;
}

[data-testid="stChatInput"] button {
    background-color: #1f6fff !important;
    color: white !important;
    border-radius: 8px !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #0d4fd6 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Study Pakistan History with AI")
st.caption("Trained on O Level Pakistan Studies History Book")

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