import streamlit as st
import tempfile
import os

from src.conversation_manager import (
    create_new_chat,
    get_all_conversations,
    switch_chat,
    delete_chat
)

from src.memory import get_session_history

from src.pdf_processor import process_pdf

from src.rag_pipeline import (
    create_vector_store,
    create_retriever
)

from src.chatbot import ask_chatbot


st.set_page_config(
    page_title="AI Conversational Assistant",
    page_icon="🤖",
    layout="wide"
)


if "active_chat" not in st.session_state:
    st.session_state.active_chat = create_new_chat()

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


with st.sidebar:

    st.title("🤖 AI Assistant")

    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.active_chat = create_new_chat()

        st.rerun()

    st.divider()

    # PDF upload
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    # Process uploaded PDF only once
    if uploaded_file is not None:

        import hashlib

        pdf_bytes = uploaded_file.getvalue()
        current_hash = hashlib.md5(pdf_bytes).hexdigest()

        if current_hash != st.session_state.pdf_hash:

            with st.spinner("Processing PDF..."):

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(pdf_bytes)
                        temp_path = temp_file.name

                    documents = process_pdf(temp_path)

                    vector_store = create_vector_store(documents)

                    st.session_state.retriever = create_retriever(
                        vector_store
                    )

                    st.session_state.pdf_hash = current_hash
                    st.session_state.pdf_name = uploaded_file.name

                    st.success("PDF processed successfully!")

                except Exception as e:

                    st.error(f"PDF processing failed: {e}")

                finally:

                    if "temp_path" in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)

        else:
            st.caption(f"Active PDF: {st.session_state.pdf_name}")

    # Remove PDF
    if st.session_state.retriever is not None:

        if st.button("Remove PDF", use_container_width=True):

            st.session_state.retriever = None
            st.session_state.pdf_hash = None
            st.session_state.pdf_name = None

            st.rerun()

    st.divider()

    # Chat history
    st.subheader("Chat History")

    conversations = get_all_conversations()

    for chat in conversations:

        session_id = chat["session_id"]
        title = chat["title"]

        col1, col2 = st.columns([4, 1])

        with col1:

            if st.button(
                title,
                key=f"chat_{session_id}",
                use_container_width=True
            ):

                st.session_state.active_chat = session_id
                st.rerun()

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{session_id}"
            ):

                delete_chat(session_id)

                if st.session_state.active_chat == session_id:
                    st.session_state.active_chat = create_new_chat()

                st.rerun()


st.title("AI Conversational Assistant")

if st.session_state.retriever is not None:

    st.caption(f"📄 PDF Mode: {st.session_state.pdf_name}")

else:

    st.caption("💬 Normal Chat Mode")


# Display existing messages
history = get_session_history(st.session_state.active_chat)

for message in history.messages:

    if message.type == "human":
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)


question = st.chat_input("Ask me anything...")

if question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = ask_chatbot(
                    question=question,
                    session_id=st.session_state.active_chat,
                    retriever=st.session_state.retriever
                )

                st.markdown(response)

            except Exception as e:

                st.error(str(e))

    st.rerun()
