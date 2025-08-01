
import os
import re
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse
from langchain_core.prompts import ChatPromptTemplate

# Define the path to your Qdrant database directory
qdrant_directory = os.path.join(os.getcwd(), "qdrant_data2")
print(qdrant_directory)

# Initialize the Qdrant client (now using a persistent path)
client = QdrantClient(path=qdrant_directory)

# List existing collections
collections = client.get_collections()
print("Existing collections:", collections)

# Define the collection name that you wish to use
collection_name = "PDF"  # Updated collection name

# Initialize the embeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")  # Ensure the same embedding model is used

# Set up the Qdrant Vector Store with hybrid capabilities
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")  # Use appropriate sparse model name

qdrant = QdrantVectorStore(
    client=client,
    collection_name="PDF",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)

# Initialize the Ollama model for generation using LangChain's OllamaLLM
ollama_model = OllamaLLM(model="deepseek-r1:7b")

# Define the prompt template using LangChain's ChatPromptTemplate
template = """Question: {question}
Answer: Let's think step by step. Here's the context that might help:
{context}
Now, based on the above context, here is the answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and the model into a chain
chain = prompt | ollama_model

# Function to perform search and then generate an answer
def perform_rag_search_and_generate(query):
    # Step 1: Perform a similarity search to retrieve relevant documents
    found_docs = qdrant.similarity_search(query, k=10)

    if found_docs:
        print("Search results:")
        context = ""
        for doc in found_docs:
            context += doc.page_content + "\n"

        # Step 2: Generate the answer by passing the query and context to the model
        answer = chain.invoke({"question": query, "context": context})

        # Remove the unwanted <think> and </think> tags, along with the content between them
        cleaned_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        print(cleaned_answer)
    else:
        print("No documents found.")

# Example search query for the RAG system
search_query = "Explain the importance of throwable class and its methods?"
perform_rag_search_and_generate(search_query)
