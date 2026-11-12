from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Initialize embedding model
embedding_model = OllamaEmbeddings(
    model="all-minilm"
)


def create_vector_store(documents):

    try:
        if not documents:
            raise ValueError("No documents provided")

        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Vector store creation failed: {e}")


def create_retriever(vector_store):

    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        return retriever

    except Exception as e:
        raise RuntimeError(f"Retriever creation failed: {e}")