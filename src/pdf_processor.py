from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(pdf_path):

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        if not pages:
            raise ValueError("No text found in PDF")

        # Split PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        documents = text_splitter.split_documents(pages)

        return documents

    except Exception as e:
        raise RuntimeError(f"PDF processing failed: {e}")