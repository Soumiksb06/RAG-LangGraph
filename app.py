# -*- coding: utf-8 -*-
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document, HumanMessage
from typing import List

# Set page config
st.set_page_config(page_title="LangGraph AI Agents", layout="wide")

# API Keys Input
st.sidebar.title("API Key Configuration")
astra_db_token = st.sidebar.text_input("Enter AstraDB Token:", type="password")
astra_db_id = st.sidebar.text_input("Enter AstraDB Database ID:")
tavily_api_key = st.sidebar.text_input("Enter Tavily API Key:", type="password")

# App Title
st.title("LangGraph AI Agents")

# Input for URLs
st.header("1. Add URLs to Populate Vectorstore")
user_urls = st.text_area("Enter URLs (comma-separated):")
urls = [url.strip() for url in user_urls.split(",") if url.strip()]

if urls:
    st.success(f"Received {len(urls)} URLs for processing.")
else:
    st.warning("Please enter at least one URL to proceed.")

# Initialize Vectorstore if API keys are provided
if astra_db_token and astra_db_id:
    import cassio

    cassio.init(token=astra_db_token, database_id=astra_db_id)

    # Load documents from URLs
    docs = []
    try:
        st.info("Fetching documents from URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        doc_list = [item for sublist in docs for item in sublist]
        st.success(f"Fetched {len(doc_list)} documents.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")

    # Split documents into chunks
    if docs:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=0)
        texts = text_splitter.split_documents(doc_list)
        st.info(f"Split documents into {len(texts)} chunks.")

        # Initialize embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        astra_vec_store = Cassandra(embedding=embeddings, table_name="ai_agents", session=None, keyspace=None)

        # Add documents to vectorstore
        astra_vec_store.add_documents(texts)
        st.success(f"Inserted {len(texts)} chunks into the vectorstore.")
else:
    st.error("Please provide AstraDB API Key and Database ID to initialize the vectorstore.")

# RAG Workflow
st.header("2. Perform Retrieval-Augmented Generation (RAG)")
question = st.text_input("Enter your question:")

if question and astra_db_token and astra_db_id:
    try:
        # Use vectorstore retriever
        retriever = astra_vec_store.as_retriever()
        response = retriever.invoke(question)

        # Display results
        st.subheader("RAG Results")
        if response:
            for i, doc in enumerate(response, 1):
                st.markdown(f"**Document {i}:**\n{doc.page_content}")
        else:
            st.warning("No results found in the vectorstore.")
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
else:
    if not question:
        st.warning("Please enter a question to proceed.")
    elif not (astra_db_token and astra_db_id):
        st.warning("Ensure AstraDB API Key and Database ID are provided.")

# Tavily Integration (Optional)
st.header("3. Tavily Search Integration (Optional)")
if tavily_api_key:
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=tavily_api_key)

    tavily_question = st.text_input("Enter your Tavily search query:")
    if tavily_question:
        try:
            results = tavily_client.search(query=tavily_question, search_depth="advanced", max_results=3)
            st.subheader("Tavily Search Results")
            for result in results:
                st.markdown(f"- **URL:** {result.url}\n- **Title:** {result.title}\n- **Snippet:** {result.snippet}")
        except Exception as e:
            st.error(f"Tavily search error: {e}")
else:
    st.info("Provide your Tavily API Key in the sidebar to enable Tavily search.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed with ❤️ using LangChain and Streamlit**")
