from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]


# ---- LangChain Groq Chat Model Example ----

# Create a chatgroq model
model = ChatGroq(model="mixtral-8x7b-32768")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from ChatGroq: {result.content}")



# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")