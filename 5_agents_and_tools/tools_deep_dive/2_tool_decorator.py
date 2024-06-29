from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


# Simple Tool with one parameter without args_schema 
# This is a basic tool that does not require an input schema 
# Use this approach for simple function that need only one parameter
@tool()
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"


# Pydantic model for tool arguments 
# Define a Pydantic model to specify the input schema for tools that need more structured input.
class ReverseStringsArgs(BaseModel):
    text: str = Field(desciption="Text to be reversed.")
    
# Tool with One Parameter using args_schema
# Use the args_schema parameter to specify the input schema usingn a Pydantic model.
@tool(args_schema=ReverseStringsArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

# Another Pydantic model for tool arguments 
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First String")
    b: str = Field(description="Second String")
    
# Tool with Two parameters using args_schema
# This tool requires multiple input parameters, so we use the args_schema to define the schema
@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenate two strings."""
    print("a", a)
    print("b", b)
    return a+b
    
    
# Create tools using the @tool decorator
# The @tool decorator simplifies the process of defining tools by handling the setup automatically
tools = [
    greet_user,
    reverse_string,
    concatenate_strings,
]

# Initialize LLM
llm = ChatGroq(model="mixtral-8x7b-32768")

# Pull the prompt template from hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
# This function sets up an agent capable of calling tools based on the provided prompt.
agent = create_tool_calling_agent(
    llm = llm,
    tools = tools,
    prompt = prompt,
)

# Create the agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors = True,
)

# Test the agent with simple queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string Niggamaru"})
print("Response for 'Reverse the string Niggamaru':", response)

response = agent_executor.invoke({"input": "Concatenate 'Yeah' and 'Baby'"})
print("Response for 'Concatenate yeah and baby':", response)