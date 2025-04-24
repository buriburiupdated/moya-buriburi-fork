"""
FAISS CPU vectorstore repositpry for Moya

An implementation of BaseVectorstoreRepository that uses the FAISS library to store and retrieve vectors
"""
from moya.vectorstore.base_vectorstore import BaseVectorstoreRepository
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

class FAISSCPUVectorstoreRepository(BaseVectorstoreRepository):
    def __init__(self, path, embeddings):
        """
        Initialize the repository with a specific path and embedding model
        """
        self.path = path
        self.embeddings = embeddings
        self.index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        self.vectorstore = None
    
    def create_vectorstore(self):
        """
        Create a new vectorstore in the specified directory
        """
        self.vectorstore = FAISS(embedding_function=self.embeddings, index=self.index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    
    def load_vectorstore(self):
        """
        Load a vectorstore from the specified directory
        """
        self.vectorstore = FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)

    def add_vector(self, chunks):
        """
        Add a new vector to the vectorstore
        """
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.path)

    def get_context(self, query, k):
        """
        Retrieve the k closest vectors to the query
        """
        results = self.vectorstore.similarity_search(query, k)
        if not results:
            return []
        return results