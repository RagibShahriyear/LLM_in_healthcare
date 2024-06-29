from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# LLM declare
model = ChatGroq(model="mixtral-8x7b-32768")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "bodybuilders", "joke_count": 3})

# Output
print(result)