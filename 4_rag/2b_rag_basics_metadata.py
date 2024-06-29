import os
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

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
query = "United States Bill of Rights Release Date"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")