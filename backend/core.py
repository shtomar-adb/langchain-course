import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeSparseVectorStore, PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieval_context(query: str):
    """Retrieve relevant documents to help answer user queries about Langchain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}") for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
        - answer: The generated answer
        - sources: List of retrieved documents
    """
    # Create the agent with retrieval tool:
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(
        model, tools=[retrieval_context], system_prompt=system_prompt
    )

    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    # Extract context documents from Toolmessage artifacts
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="What are deep agents?")
    print(result)