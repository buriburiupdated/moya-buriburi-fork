"""
Chroma vectorstore repository for Moya

An implementation of BaseVectorstoreRepository that uses the Chroma library to store and retrieve vectors
"""
from moya.vectorstore.base_vectorstore import BaseVectorstoreRepository
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from typing import List

class ChromaVectorstoreRepository(BaseVectorstoreRepository):
    def __init__(self, path, embeddings):
        """
        Initialize the repository with a specific path and embedding model
        """
        self.embedding = embeddings
        self.path = path
        self.vectorstore = None

    def create_vectorstore(self) -> None:
        """
        Create a new vectorstore in the specified directory
        """
        self.vectorstore = Chroma(embedding_function=self.embedding, persist_directory=self.path)

    def load_vectorstore(self, directory: str) -> None:
        """
        Load a vectorstore from the specified directory
        """
        self.vectorstore = Chroma(embedding_function=self.embedding, persist_directory=self.path)

    def add_vector(self, chunks: List[Document]) -> None:
        """
        Add a new vector to the vectorstore
        """
        self.vectorstore.add_documents(chunks)
    
    def get_context(self, query: str, k: int) -> List[Document]:
        """
        Retrieve the k closest vectors to the query
        """
        results = self.vectorstore.similarity_search(query,k)
        if not results:
            return []
        return results