import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

CHROMA_PERSIST_DIRECTORY = "chroma_db"

class ChunkVectorStore:
    def __init__(self) -> None:
        pass

    def split_into_chunks(self, file_path: str):
        doc = PyPDFLoader(file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunks = text_splitter.split_documents(doc)
        chunks = filter_complex_metadata(chunks)
        return chunks

    def store_to_vector_database(self, chunks):
        try:
            Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=FastEmbedEmbeddings()).delete_collection()
            print("Existing Chroma collection deleted.")
        except ValueError as e:
            if "Collection not found" in str(e):
                print("No existing Chroma collection found. Creating a new one.")
            else:
                print(f"An error occurred during collection deletion: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during collection deletion: {e}")

        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        return Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )