from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGroq(model="mixtral-8x7b-32768")

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)