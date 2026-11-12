# AI Conversational Assistant with RAG and Memory

## Project Overview
An AI-powered conversational assistant that works as a normal chatbot and automatically switches to Retrieval-Augmented Generation (RAG) when a user uploads a PDF document. It also maintains temporary conversation history, allowing users to create, switch between, and delete chats.

## Key Features
- **Normal Chatbot:** Answers general questions using a Groq-hosted Qwen LLM.
- **PDF-Based RAG:** Extracts, chunks, and indexes uploaded PDF documents using FAISS.
- **Context-Aware Retrieval:** Understands follow-up questions using conversation history.
- **Conversation Memory:** Maintains separate chat histories for different conversations.
- **Chat Management:** Create new chats, switch between conversations, and delete chats.
- **Dynamic Mode Switching:** Automatically switches between normal chat and PDF-based RAG.
- **Temporary Storage:** Conversations are maintained in memory and cleared when the application process restarts.

## Tech Stack
- Python
- LangChain
- Groq (Qwen LLM)
- Ollama Embeddings (`all-minilm`)
- FAISS
- Streamlit
- PyPDFLoader
- LangSmith (optional monitoring)

## Project Structure
```text
conversational_ai_assistant/
│
├── notebooks/
├── src/
│   ├── llm_config.py
│   ├── pdf_processor.py
│   ├── rag_pipeline.py
│   ├── memory.py
│   ├── conversation_manager.py
│   └── chatbot.py
│
├── data/
├── app.py
├── .env
├── .gitignore
└── requirements.txt
```
**2. Create a virtual environment**
```bash
python -m venv chatbot
```

**3. Activate the environment (Windows)**
```bash
chatbot\Scripts\activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**6. Start Ollama**

Install Ollama and pull the embedding model:
```bash
ollama pull all-minilm
```

**7. Run the application**
```bash
streamlit run app.py
```

## How It Works
1. Without a PDF, the assistant answers questions using the normal LLM pipeline.
2. When a PDF is uploaded, it extracts and splits the document into chunks.
3. Ollama generates embeddings, which are stored in a FAISS vector store.
4. Relevant chunks are retrieved for user questions.
5. The Groq-hosted Qwen model generates answers using the retrieved context.
6. Conversation memory helps the assistant understand follow-up questions.

## Future Improvements
- Persistent conversation storage.
- Support for multiple PDF documents.
- Chat export functionality.
- User authentication and session isolation.
- Improved document retrieval and response evaluation.

## Author
**Bhabani Sankar Sahoo**

Generative AI | Machine Learning | Python | LangChain