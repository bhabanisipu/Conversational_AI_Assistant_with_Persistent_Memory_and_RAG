from pathlib import Path
from src.pdf_processor import process_pdf
from src.rag_pipeline import (
    create_vector_store,
    create_retriever
)

pdf_path = (
    Path(__file__).resolve().parent
    / "data"
    / "12 GenAI Concepts Everyone Thinks They Know but don't .pdf"
)

# Process PDF
documents = process_pdf(pdf_path)

# Create vector store
vector_store = create_vector_store(documents)

# Create retriever
retriever = create_retriever(vector_store)

# Test retrieval
results = retriever.invoke(
    "What is the main topic of this document?"
)

for doc in results:
    print(doc.page_content)
    print("-------------------")