import os
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

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

# Define the user's question
query = "How can I learn more about LangChain?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# LLM model
model = ChatGroq(model="mixtral-8x7b-32768")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)