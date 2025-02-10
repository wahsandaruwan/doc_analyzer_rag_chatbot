import os
from chunk_vector_store import ChunkVectorStore as cvs
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import streamlit as st

CHROMA_PERSIST_DIRECTORY = "chroma_db"  # Consistent definition

class Rag:
    def __init__(self) -> None:
        self.csv_obj = cvs()
        # self.prompt = PromptTemplate.from_template(
        #     """
        #     <s> [INST] You are a helpful assistant designed for question-answering tasks. Use the provided context to answer the user's question.

        #     **Instructions:**

        #     If the context contains sufficient information to confidently answer the question, provide a concise answer (maximum 10 sentences). If the context does *not* contain sufficient information (less than 50% of relevant information is present), respond with: "Insufficient information to provide a complete answer. But here's what I found from the knowledge base:" provide the retrieved information (maximum 10 sentences).
        #     [/INST] </s>
        #     [INST] Question: {question}
        #     Context: {context}
        #     Answer: [/INST]
        #     """
        # )
        self.prompt = PromptTemplate.from_template(
            """
            Use the following context to answer the question. If the context doesn't contain the answer, say "I don't know".

            Context: {context}
            Question: {question}
            Answer:
            """
        )
        self.model = ChatOllama(model="Llama3.1")
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def set_retriever(self):
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5,
                },
            )
        else:
            self.retriever = None

    def augment(self):
        if self.retriever:
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
        else:
            self.chain = None

    def ask(self, query: str):
        if not self.chain:
            return "Please upload a PDF file for context"
        return self.chain.invoke(query)

    def feed(self, file_path: str):
        with st.spinner("Processing file..."):
            print("Feeding new file. Clearing existing data.")
            self.clear()
            chunks = self.csv_obj.split_into_chunks(file_path)
            self.vector_store = self.csv_obj.store_to_vector_database(chunks)
            self.set_retriever()
            self.augment()
            print("New vector store created and ready.")

    def clear(self):
        print("Clearing existing vector store, retriever, and chain.")
        if self.vector_store:
            try:
                Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=FastEmbedEmbeddings()).delete_collection()
                print("Existing Chroma collection deleted.")
            except ValueError as e:
                if "Collection not found" in str(e):
                    print("No existing Chroma collection found.")
                else:
                    print(f"An error occurred during collection deletion: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during collection deletion: {e}")

        self.vector_store = None
        self.chain = None
        self.retriever = None
        print("Cleared successfully.")