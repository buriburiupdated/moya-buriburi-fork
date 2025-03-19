"""
Vector search tool implementation for Moya.
Provides RAG capabilities by searching through vector databases.
"""

import json
from typing import Dict, List, Any, Optional
from moya.tools.base_tool import BaseTool

class VectorSearchTool:
    """Tools for vector database search capabilities."""
    
    @staticmethod
    def search_vectorstore(query: str, vector_store: Any, k: int = 5) -> str:
        """
        Search a vector database for relevant documents based on semantic similarity.
        
        Args:
            query: The search query text
            collection_name: Name of the vector collection to search in
            k: Number of results to return
            
        Returns:
            JSON string containing search results
        """
        try:
            # Get the appropriate vector store
            
            # Search for relevant documents
            results = vector_store.get_context(query, k)

            formatted_results = []
            for i, doc in enumerate(results):
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None)
                })
            
            return json.dumps({
                "query": query,
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": query,
            })
    
    @staticmethod
    def configure_vector_search_tools(tool_registry) -> None:
        """
        Configure vector search tools and register them with the tool registry.
        
        Args:
            tool_registry: The tool registry to register tools with.
        """
        tool_registry.register_tool(
            BaseTool(
                name="VectorSearchTool",
                function=VectorSearchTool.search_vectorstore,
                description="Search a vector database for semantically similar content",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                        "required": True
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the vector collection to search (default: 'faiss-index')",
                        "required": False
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False
                    }
                }
            )
        )