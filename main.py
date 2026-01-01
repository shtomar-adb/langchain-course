from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown")) 
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="Langchain Documentation Helper", layout="centered")
st.title("Langchain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask me anything about LangChain docs")
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer retruned)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)

        


# import os

# from operator import itemgetter
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()

# print("Initializing components...")

# embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-small", dimensions=1024)
# llm = ChatOpenAI()

# vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embedding)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# prompt_template = ChatPromptTemplate.from_template(
#     """Answer the question based only on the following context:
#     {context}

#     Question: {question}

#     Provide a detailed answer:"""
# )

# def format_docs(docs):
#     """Format retrieved documents into a single string"""
#     return "\n\n".join(doc.page_content for doc in docs)


# def retrieval_chain_without_lcel(query: str):
#     """



#     Simple retrieval chain without LCEL.


#     Manually retrieves documents, formats them, and generates a response.





#     Limitations:


#     - Manual step-by-step execution


#     - No built-in streaming support


#     - No async support without additional code


#     - Harder to compose with other chains


#     - More verbose and error-prone


#     """

#     docs = retriever.invoke(query)
#     context = format_docs(docs)
#     messages = prompt_template.format_messages(context=context, question=query)

#     response = llm.invoke(messages)

#     return response.content


# def create_retrieval_chain_with_lcel():
#     """



#     Create a retrieval chain using LCEL (LangChain Expression Language).


#     Returns a chain that can be invoked with {"question": "..."}





#     Advantages over non-LCEL approach:


#     - Declarative and composable: Easy to chain operations with pipe operator (|)


#     - Built-in streaming: chain.stream() works out of the box


#     - Built-in async: chain.ainvoke() and chain.astream() available


#     - Batch processing: chain.batch() for multiple inputs


#     - Type safety: Better integration with LangChain's type system


#     - Less code: More concise and readable


#     - Reusable: Chain can be saved, shared, and composed with other chains


#     - Better debugging: LangChain provides better observability tools


#     """
#     retrieval_chain = (
#         RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs) | prompt_template | llm | StrOutputParser()
#     )

#     return retrieval_chain


# if __name__ == "__main__":
#     print("retrieving....")

#     query = "why people are not reading anymore?"

#     # Without LCEL
#     print("\n" + "=" * 70)
#     print("Implementation without LCEL")
#     print("=" * 70)
#     result_without_lcel = retrieval_chain_without_lcel(query)
#     print("\n Answer:")
#     print(result_without_lcel)

#     # Option 2: Use implementation WITH LCEL (Better Approach)
#     print("\n" + "=" * 70)
#     print("Implementation with LCEL")
#     print("=" * 70)
#     print("Why LCEL is better:")



#     print("- More concise and declarative")


#     print("- Built-in streaming: chain.stream()")


#     print("- Built-in async: chain.ainvoke()")


#     print("- Easy to compose with other chains")


#     print("- Better for production use")


#     print("=" * 70)

#     chain_with_lcel = create_retrieval_chain_with_lcel()
#     result_with_lcel = chain_with_lcel.invoke({"question": query})
#     print("\n Answer:")
#     print(result_with_lcel)