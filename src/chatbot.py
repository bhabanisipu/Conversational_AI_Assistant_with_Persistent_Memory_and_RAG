from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.llm_config import llm
from src.memory import get_session_history
from src.conversation_manager import update_chat_title


normal_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Answer the user's questions clearly and accurately.
        Maintain context from the previous conversation."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# Normal chatbot chain
normal_chain = normal_prompt | llm | StrOutputParser()


# Add conversation memory
normal_chat_chain = RunnableWithMessageHistory(
    normal_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given the conversation history and the latest
        user question, rewrite the latest question as a
        standalone question.

        Rules:
        1. Do not answer the question.
        2. Understand references such as it, they, this and that.
        3. If the question is already clear, return it unchanged.
        4. Return only the standalone question."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

contextualize_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

        Answer the user's question using the provided
        document context and conversation history.

        Rules:
        1. Use document context as the primary source.
        2. Use conversation history to understand follow-up questions.
        3. Do not invent information.
        4. If the answer is unavailable in the document,
           clearly say that it is not available in the document.
        5. Give a clear and concise answer.

        Document Context:
        {context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# RAG chain
rag_chain = rag_prompt | llm | StrOutputParser()


# Add conversation memory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


def ask_chatbot(question, session_id, retriever=None):

    try:

        # Update conversation title
        update_chat_title(session_id, question)


        if retriever is None:

            response = normal_chat_chain.invoke(
                {"question": question},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            return response

        history = get_session_history(session_id)

        # Rewrite follow-up question
        if history.messages:

            standalone_question = contextualize_chain.invoke({
                "chat_history": history.messages,
                "question": question
            })

        else:
            standalone_question = question

        # Retrieve relevant document chunks
        retrieved_docs = retriever.invoke(standalone_question)

        # Combine retrieved document chunks
        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # Generate answer using RAG
        response = conversational_rag_chain.invoke(
            {
                "context": context,
                "question": question
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        return response

    except Exception as e:

        raise RuntimeError(f"Chatbot failed: {e}")
