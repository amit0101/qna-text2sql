import streamlit as st
from models.llm import answer_question
import uuid


def main():
    # Initialize session state for managing chat history
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Generate a unique session ID

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Streamlit UI
    st.title("RAG App")
    st.write("Ask me anything based on the provided PDFs!")

    # Input box for user question
    question = st.text_input("Enter your question:")

    if question:
        # Get answer from the RAG model
        answer, count = answer_question(question, st.session_state.session_id)

        # Update chat history
        st.session_state.chat_history.append({"role": "user", "message": question})
        st.session_state.chat_history.append({"role": "assistant", "message": answer})

        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.write(f"**You**: {chat['message']}", unsafe_allow_html=True)
            else:
                st.write(f"**Assistant**: {chat['message']}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
