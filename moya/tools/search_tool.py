"""
Search tool implementation for MOYA.
Provides web search capabilities for agents.
"""

import os
import json
import requests
from typing import Dict, List, Optional
from moya.tools.base_tool import BaseTool

class SearchTool:
    """Tools for web search capabilities."""
    
    @staticmethod
    def search_web(query: str, num_results: int = 5) -> str:
        """
        Search the web using SerpAPI.
        
        Args:
            query (str): The search query
            num_results (int, optional): Number of results to return. Defaults to 5.
            
        Returns:
            str: JSON string with search results
        """
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return json.dumps({
                "error": "SERPAPI_API_KEY environment variable not set. Please set it to use web search."
            })
            
        try:
            # Use SerpAPI for search results
            params = {
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "num": str(num_results)
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            results = response.json()
            
            # Extract organic results
            if "organic_results" in results:
                formatted_results = []
                for result in results["organic_results"][:num_results]:
                    formatted_results.append({
                        "title": result.get("title", "No Title"),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "No Description")
                    })
                
                return json.dumps({
                    "query": query,
                    "results": formatted_results
                }, indent=2)
            else:
                return json.dumps({
                    "error": "No results found",
                    "query": query
                })
                
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": query
            })
    
    @staticmethod
    def search_web_free(query: str, num_results: int = 5) -> str:
        """Search the web using DuckDuckGo (no API key required)."""
        print("\n" + "="*50)
        print(f"[DEBUG] FreeWebSearchTool called with query: '{query}'")
        print("[DEBUG] Sending request to DuckDuckGo...")
        
        try:
            # Use DuckDuckGo HTML endpoint
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Encode the query
            from urllib.parse import quote_plus
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            print(f"[DEBUG] Request URL: {url}")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            print(f"[DEBUG] Response status code: {response.status_code}")
            
            # Extract results using basic HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='result')
            print(f"[DEBUG] Found {len(results)} raw results")
            
            formatted_results = []
            for i, result in enumerate(results[:num_results]):
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                title = title_elem.text.strip() if title_elem else "No Title"
                link = title_elem['href'] if title_elem and 'href' in title_elem.attrs else ""
                snippet = snippet_elem.text.strip() if snippet_elem else "No Description"
                
                formatted_results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })
                print(f"[DEBUG] Result {i+1}: {title[:40]}...")
            
            result_json = json.dumps({
                "query": query,
                "results": formatted_results
            }, indent=2)
            print(f"[DEBUG] Search completed successfully with {len(formatted_results)} formatted results")
            print("="*50)
            return result_json
                
        except Exception as e:
            print(f"[DEBUG] Search failed with error: {str(e)}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            print("="*50)
            return json.dumps({
                "error": str(e),
                "query": query
            })
    
    @staticmethod
    def configure_search_tools(tool_registry) -> None:
        """
        Configure search tools and register them with the tool registry.
        
        Args:
            tool_registry: The tool registry to register tools with.
        """
        # Register paid search tool (using SerpAPI)
        tool_registry.register_tool(
            BaseTool(
                name="WebSearchTool",
                function=SearchTool.search_web,
                description="Search the web for information using SerpAPI",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                        "required": True
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False
                    }
                }
            )
        )
        
        # Register free search tool (using DuckDuckGo)
        tool_registry.register_tool(
            BaseTool(
                name="FreeWebSearchTool",
                function=SearchTool.search_web_free,
                description="Search the web for information using DuckDuckGo (no API key required)",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                        "required": True
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False
                    }
                }
            )
        )