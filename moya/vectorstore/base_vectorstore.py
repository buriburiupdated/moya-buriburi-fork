"""
Base repository for vectorstores in Moya

Defines an abstract class describing how to store and retrieve vectors in multiple different databases
"""

import abc
from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader, JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BaseVectorstoreRepository(abc.ABC):
    """
    Abstract interface for storing and retrieving vectors
    """ 

    @abc.abstractmethod
    def create_vectorstore(self, direcotry: str) -> None:
        """
        Create a new vectorstore in the specified directory
        """
        pass
    
    @abc.abstractmethod
    def add_vector(self, chunks: List[Document]) -> None:
        """
        Add a new vector to the vectorstore
        """
        pass

    @abc.abstractmethod
    def get_context(self, query: str, k: int) -> List[Document]:
        """
        Retrieve the k closest vectors to the query
        """
        pass

    @abc.abstractmethod
    def load_vectorstore(self, directory: str) -> None:
        """
        Load a vectorstore from the specified directory
        """
        pass

    def splitter(self, docs, chunk_size: int, chunk_overlap: int) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        self.add_vector(splits)


    def load_file(self, path: str)-> None:
        """
        Load a file into vectorstore
        """
        if not path:
            raise ValueError("path is required")
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".html"):
            loader = BSHTMLLoader(path)
        elif path.endswith(".md"):
            loader = TextLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".json"):
            loader = JSONLoader(file_path=path, text_content=False)
        elif path.endswith(".csv"):
            loader = CSVLoader(file_path=path)
        else:
            return
        docs = loader.load()
        self.splitter(docs, 1000, 200)