import uuid
from datetime import datetime

from src.memory import (
    get_session_history,
    delete_session_history
)

from src.llm_config import llm


# Store conversation details temporarily
conversations = {}


def create_new_chat():
    try:
        session_id = str(uuid.uuid4())

        # Create an empty chat history
        get_session_history(session_id)

        conversations[session_id] = {
            "title": "New Chat",
            "created_at": datetime.now()
        }

        return session_id

    except Exception as e:
        raise RuntimeError(f"Chat creation failed: {e}")


def get_all_conversations():
    try:
        chat_list = []

        for session_id, details in conversations.items():
            chat_list.append({
                "session_id": session_id,
                "title": details["title"],
                "created_at": details["created_at"]
            })

        # Show latest conversations first
        return sorted(
            chat_list,
            key=lambda x: x["created_at"],
            reverse=True
        )

    except Exception as e:
        raise RuntimeError(f"Unable to fetch conversations: {e}")


def switch_chat(session_id):
    if session_id not in conversations:
        raise ValueError("Conversation not found")

    return get_session_history(session_id)


def delete_chat(session_id):
    if session_id not in conversations:
        raise ValueError("Conversation not found")

    del conversations[session_id]
    delete_session_history(session_id)

    return True


def generate_chat_title(question):
    try:
        prompt = f"""
        Generate a short title for this conversation.

        User question: {question}

        Rules:
        1. Maximum 5 words.
        2. Do not use quotation marks.
        3. Return only the title.
        """

        response = llm.invoke(prompt)

        title = response.content.strip().strip('"').strip("'")

        return title if title else question[:40]

    except Exception:
        return question[:40]


def update_chat_title(session_id, question):
    if session_id not in conversations:
        raise ValueError("Conversation not found")

    if conversations[session_id]["title"] == "New Chat":

        title = generate_chat_title(question)

        conversations[session_id]["title"] = title

    return conversations[session_id]["title"]
