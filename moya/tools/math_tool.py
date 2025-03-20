"""
Enhanced Math tool implementation for MOYA.
Provides comprehensive mathematical capabilities for agents.
"""

from sympy import symbols, Eq, solve, diff, integrate, limit, series, Matrix, sympify
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, Any, List
from moya.tools.base_tool import BaseTool

class MathTool:
    """Tools for comprehensive mathematical capabilities."""
    
    @staticmethod
    def solve_equation(equation: str, variable: str) -> Dict[str, Any]:
        """
        Solve a symbolic math equation for a given variable.
        
        Args:
            equation: The equation to solve
            variable: The variable to solve for
            
        Returns:
            A dictionary containing the solution
        """
        try:
            var = symbols(variable)
            # Support both formats: "x**2 = 4" and "x**2 - 4"
            if "=" in equation:
                left, right = equation.split("=", 1)
                eq = Eq(parse_expr(left.strip()), parse_expr(right.strip()))
            else:
                eq = Eq(parse_expr(equation.strip()), 0)
            solution = solve(eq, var)
            return {"solution": str(solution)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def differentiate(expression: str, variable: str, order: int = 1) -> Dict[str, Any]:
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            expression: The mathematical expression to differentiate
            variable: The variable to differentiate with respect to
            order: The order of differentiation (default=1)
            
        Returns:
            A dictionary containing the derivative
        """
        try:
            var = symbols(variable)
            expr = parse_expr(expression)
            result = diff(expr, var, order)
            return {"derivative": str(result)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def integrate_expression(expression: str, variable: str) -> Dict[str, Any]:
        """
        Integrate an expression with respect to a variable.
        
        Args:
            expression: The mathematical expression to integrate
            variable: The variable to integrate with respect to
            
        Returns:
            A dictionary containing the integral
        """
        try:
            var = symbols(variable)
            expr = parse_expr(expression)
            result = integrate(expr, var)
            return {"integral": str(result)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def evaluate_expression(expression: str, values: str) -> Dict[str, Any]:
        """
        Evaluate an expression with specified variable values.
        
        Args:
            expression: The mathematical expression to evaluate
            values: A string representing variable assignments (e.g., "x=2,y=3")
            
        Returns:
            A dictionary containing the evaluated result
        """
        try:
            expr = parse_expr(expression)
            # Parse the values string into a dictionary
            value_dict = {}
            for assignment in values.split(","):
                var, val = assignment.split("=")
                value_dict[var.strip()] = float(val.strip())
            
            result = expr.subs(value_dict)
            return {"result": str(result)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def solve_system(equations: str, variables: str) -> Dict[str, Any]:
        """
        Solve a system of linear equations.
        
        Args:
            equations: A semicolon-separated list of equations
            variables: A comma-separated list of variables
            
        Returns:
            A dictionary containing the solutions
        """
        try:
            var_symbols = symbols(variables)
            if not isinstance(var_symbols, tuple):
                var_symbols = (var_symbols,)
                
            eqs = []
            for eq_str in equations.split(";"):
                if "=" in eq_str:
                    left, right = eq_str.split("=", 1)
                    eq = Eq(parse_expr(left.strip()), parse_expr(right.strip()))
                else:
                    eq = Eq(parse_expr(eq_str.strip()), 0)
                eqs.append(eq)
                
            solution = solve(eqs, var_symbols)
            return {"solution": str(solution)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def compute_limit(expression: str, variable: str, point: str) -> Dict[str, Any]:
        """
        Compute the limit of an expression as variable approaches point.
        
        Args:
            expression: The mathematical expression
            variable: The variable in the limit
            point: The point the variable approaches
            
        Returns:
            A dictionary containing the limit
        """
        try:
            var = symbols(variable)
            expr = parse_expr(expression)
            point_val = parse_expr(point)
            result = limit(expr, var, point_val)
            return {"limit": str(result)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def expand_series(expression: str, variable: str, point: str, order: int) -> Dict[str, Any]:
        """
        Expand an expression as a series around a point.
        
        Args:
            expression: The mathematical expression
            variable: The variable for expansion
            point: The point around which to expand
            order: The order of the series expansion
            
        Returns:
            A dictionary containing the series expansion
        """
        try:
            var = symbols(variable)
            expr = parse_expr(expression)
            point_val = parse_expr(point)
            result = series(expr, var, point_val, order)
            return {"series": str(result)}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def configure_math_tools(tool_registry) -> None:
        """
        Configure math tools and register them with the tool registry.
        
        Args:
            tool_registry: The tool registry to register tools with.
        """
        # Register equation solver
        tool_registry.register_tool(
            BaseTool(
                name="SolveEquation",
                function=MathTool.solve_equation,
                description="Solve symbolic math equations for a variable",
                parameters={
                    "equation": {
                        "type": "string",
                        "description": "The equation to solve (e.g., 'x**2 = 4' or 'x**2 - 4 = 0')",
                        "required": True
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to solve for (e.g., 'x')",
                        "required": True
                    }
                }
            )
        )
        
        # Register differentiation tool
        tool_registry.register_tool(
            BaseTool(
                name="Differentiate",
                function=MathTool.differentiate,
                description="Differentiate a mathematical expression",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The expression to differentiate (e.g., 'x**2 + 3*x')",
                        "required": True
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to differentiate with respect to (e.g., 'x')",
                        "required": True
                    },
                    "order": {
                        "type": "integer",
                        "description": "The order of differentiation (default=1)",
                        "required": False
                    }
                }
            )
        )
        
        # Register integration tool
        tool_registry.register_tool(
            BaseTool(
                name="Integrate",
                function=MathTool.integrate_expression,
                description="Integrate a mathematical expression",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The expression to integrate (e.g., 'x**2 + 3*x')",
                        "required": True
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable to integrate with respect to (e.g., 'x')",
                        "required": True
                    }
                }
            )
        )
        
        # Register evaluation tool
        tool_registry.register_tool(
            BaseTool(
                name="EvaluateExpression",
                function=MathTool.evaluate_expression,
                description="Evaluate a mathematical expression with specific values",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The expression to evaluate (e.g., 'x**2 + y')",
                        "required": True
                    },
                    "values": {
                        "type": "string",
                        "description": "Variable assignments (e.g., 'x=2,y=3')",
                        "required": True
                    }
                }
            )
        )
        
        # Register system solver tool
        tool_registry.register_tool(
            BaseTool(
                name="SolveSystem",
                function=MathTool.solve_system,
                description="Solve a system of linear equations",
                parameters={
                    "equations": {
                        "type": "string",
                        "description": "Semicolon-separated list of equations (e.g., 'x + y = 3; 2*x - y = 1')",
                        "required": True
                    },
                    "variables": {
                        "type": "string",
                        "description": "Comma-separated list of variables (e.g., 'x,y')",
                        "required": True
                    }
                }
            )
        )
        
        # Register limit tool
        tool_registry.register_tool(
            BaseTool(
                name="ComputeLimit",
                function=MathTool.compute_limit,
                description="Compute the limit of an expression",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The expression to find the limit of (e.g., 'sin(x)/x')",
                        "required": True
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable in the limit (e.g., 'x')",
                        "required": True
                    },
                    "point": {
                        "type": "string",
                        "description": "The point the variable approaches (e.g., '0')",
                        "required": True
                    }
                }
            )
        )
        
        # Register series expansion tool
        tool_registry.register_tool(
            BaseTool(
                name="ExpandSeries",
                function=MathTool.expand_series,
                description="Expand an expression as a series around a point",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The expression to expand (e.g., 'exp(x)')",
                        "required": True
                    },
                    "variable": {
                        "type": "string",
                        "description": "The variable for expansion (e.g., 'x')",
                        "required": True
                    },
                    "point": {
                        "type": "string",
                        "description": "The point around which to expand (e.g., '0')",
                        "required": True
                    },
                    "order": {
                        "type": "integer",
                        "description": "The order of the series expansion (e.g., 4)",
                        "required": True
                    }
                }
            )
        )