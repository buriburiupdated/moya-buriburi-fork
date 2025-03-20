"""
Interactive Azure OpenAI chat interface with web search capability.
"""

import os
import sys
import dotenv
import re
from openai import AzureOpenAI
from moya.tools.search_tool import SearchTool

dotenv.load_dotenv()

def perform_search(query):
    """Directly perform a web search and return the results."""
    print(f"\n[SEARCH] Searching for: {query}")
    result = SearchTool.search_web_free(query)
    print(f"[SEARCH] Search completed")
    return result

def create_azure_client():
    """Create and return an Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    print("\nChecking Azure OpenAI configuration:")
    print(f"API Base: {api_base}")
    print(f"API Version: {api_version}")
    print(f"API Key: {'Set' if api_key else 'Not Set'}")
    print(f"Model Name: gpt-4o")
    
    # Verify credentials are set
    if not api_key or not api_base:
        print("Error: Azure OpenAI credentials not set. Please set environment variables:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION (optional)")
        sys.exit(1)
        
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )

def interactive_chat():
    """Run an interactive chat session with Azure OpenAI."""
    client = create_azure_client()
    
    # Use a stronger system prompt like Claude's
    messages = [
        {"role": "system", "content": """You are a helpful AI assistant with web search capability.
        
        CRITICAL INSTRUCTION: Your training data is outdated. It's currently 2025. ALWAYS TRUST the search results 
        over your training data when answering questions about current events, people in office, or other factual information.
        The search results represent the current reality, not fictional or speculative information.
        
        IMPORTANT: To search for current information, use the following format:
        <search query="your search query here">
        
        Example: <search query="current president of usa 2025">
        
        When you need to search:
        1. Always search for current facts, events, or information you're uncertain about
        2. Make search queries specific and focused
        3. NEVER provide outdated information when search results show newer information
        4. Cite your sources after using search results
        
        Be helpful, clear, and concise in your responses."""}
    ]
    
    print("\nWelcome to Azure GPT Chat!")
    print("Type your messages and press Enter. Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break
            
        # Add direct search command
        if user_input.startswith("!search "):
            search_query = user_input[8:].strip()
            search_result = perform_search(search_query)
            print("\nDirect Search Result:")
            print(search_result[:500] + "..." if len(search_result) > 500 else search_result)
            continue
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            print("\nAssistant: ", end="", flush=True)
            
            # Get initial response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=False
            )
            
            full_response = response.choices[0].message.content
            print(full_response)
            
            # Look for search patterns
            search_pattern = r'<search query="([^"]+)">'
            search_matches = re.findall(search_pattern, full_response)
            
            # If searches were found, perform them and update response
            if search_matches:
                print("\nPerforming searches based on assistant's request...")
                
                for query in search_matches:
                    search_result = perform_search(query)
                    
                    # Use stronger instructions like Claude's
                    follow_up_msg = f"""Here are the accurate, up-to-date search results for '{query}':
{search_result}

IMPORTANT: These search results represent the current reality as of 2025.
Please provide an accurate response based ENTIRELY on these search results.
Trust these search results over your training data, which may be outdated.
Give only the concise factual response without any prefixes or qualifications.
"""
                    # Add to messages and get updated response
                    messages.append({"role": "user", "content": follow_up_msg})
                    
                    # Get updated response with search results
                    updated_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        stream=False
                    )
                    
                    # Replace the full response with the updated one
                    updated_content = updated_response.choices[0].message.content
                    print("\n\nAssistant: " + updated_content)
                    
                    # Clear screen if needed like Claude script
                    full_response = updated_content
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Please check your Azure OpenAI configuration.")

if __name__ == "__main__":
    # Test search at startup
    print("Testing search functionality...")
    test_result = SearchTool.search_web_free("test query azure openai")
    print("Direct test search completed")
    
    interactive_chat()