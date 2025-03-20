"""
Dynamic Tool Registrar for Moya

A utility for dynamically registering custom functions as Moya tools.
"""

from typing import Any, Callable, Dict, List, Optional
from moya.tools.tool_registry import ToolRegistry
from moya.tools.base_tool import BaseTool


class DynamicToolRegistrar:
    """
    Provides methods to dynamically register custom functions as tools at runtime
    """
    
    @staticmethod
    def register_function_as_tool(
        tool_registry: ToolRegistry,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> BaseTool:
        """
        Register a custom function as a tool in the registry.
        
        Parameters:
        - tool_registry: The ToolRegistry to register the tool in
        - function: The function to register as a tool
        - name: Optional tool name (defaults to function name if not provided)
        - description: Optional tool description (defaults to function docstring)
        - parameters: Optional parameters information (extracted from docstring if not provided)
        
        Returns:
        - The created BaseTool instance
        """
        tool_name = name or function.__name__
        tool = BaseTool(
            name=tool_name,
            description=description,
            function=function,
            parameters=parameters
        )
        tool_registry.register_tool(tool)
        return tool
    
    @staticmethod
    def register_functions(
        tool_registry: ToolRegistry,
        functions: List[Callable],
        names: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        """
        Register multiple functions as tools in one go.
        
        Parameters:
        - tool_registry: The ToolRegistry to register tools in
        - functions: List of functions to register
        - names: Optional list of tool names (will use function names if not provided)
        - descriptions: Optional list of tool descriptions (will use docstrings if not provided)
        
        Returns:
        - List of created BaseTool instances
        """
        tools = []
        for i, function in enumerate(functions):
            name = names[i] if names and i < len(names) else None
            desc = descriptions[i] if descriptions and i < len(descriptions) else None
            tool = DynamicToolRegistrar.register_function_as_tool(
                tool_registry=tool_registry,
                function=function,
                name=name,
                description=desc
            )
            tools.append(tool)
        return tools
    
    @staticmethod
    def register_from_code(
        tool_registry: ToolRegistry,
        function_code: str,
        tool_name: str,
        description: str,
    ) -> BaseTool:
        """
        Register a new tool defined by a Python function code string.
        
        Parameters:
        - tool_registry: The ToolRegistry to register the tool in
        - function_code: Python code for the function to register
        - tool_name: Name for the new tool
        - description: Description of what the tool does
        
        Returns:
        - The created BaseTool instance
        """
        try:
            # Create a namespace for the function
            namespace = {}
            
            # Execute the function code in the namespace
            exec(function_code, globals(), namespace)
            
            # Find the function in the namespace
            function_name = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('__'):
                    function_name = name
            
            if not function_name:
                raise ValueError("No function defined in the provided code.")
            
            # Register the function as a tool
            function = namespace[function_name]
            return DynamicToolRegistrar.register_function_as_tool(
                tool_registry=tool_registry,
                function=function,
                name=tool_name,
                description=description
            )
        except Exception as e:
            raise ValueError(f"Error registering tool from code: {str(e)}")
    
    @staticmethod
    def configure_dynamic_tool_registrar(tool_registry: ToolRegistry) -> None:
        """
        Configure the tool registry with tools for dynamic tool registration.
        
        Parameters:
        - tool_registry: The ToolRegistry to configure
        """
        def register_tool_by_code(function_code: str, tool_name: str, description: str) -> str:
            """
            Register a new tool defined by a Python function code string.
            
            Parameters:
            - function_code: Python code for the function to register
            - tool_name: Name for the new tool
            - description: Description of what the tool does
            
            Returns:
            - Confirmation message
            """
            try:
                DynamicToolRegistrar.register_from_code(
                    tool_registry=tool_registry,
                    function_code=function_code,
                    tool_name=tool_name,
                    description=description
                )
                return f"Tool '{tool_name}' has been successfully registered and is now available."
            except Exception as e:
                return f"Error registering tool: {str(e)}"
        
        # Register the tool registrar as a tool
        tool_registry.register_tool(BaseTool(
            name="RegisterTool",
            description="Register a new custom tool with a Python function code",
            function=register_tool_by_code
        ))