import os
import torch
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# 1. Load environment variables from .env file
load_dotenv()

# 2. Load the existing chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Check if the chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )


# Define the embedding model

device = torch.device("mps")
modelPath = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device':device}
encode_kwargs = {'normalize_embeddings': False,
                'batch_size': 32}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs, # Pass the encoding options
    show_progress = True
)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3},
)

# Create a chat model
llm = ChatGroq(model="mixtral-8x7b-32768")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt 
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions 
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Create a chain to combine documents for question answering 
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the retrieval and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


# Set-up react agent with document store retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    )
]


# Create a react agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
