import streamlit as st
import cassio
from typing import List, Literal
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_groq import ChatGroq
from langchain.schema import Document, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from tavily import TavilyClient
from langgraph.graph import END, StateGraph, START

# Define the RouteQuery class
class RouteQuery(BaseModel):
    datasource: Literal['vectorstore', 'wikipedia_search', 'tavily_search', 'default'] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia, tavily search, a vectorstore, or default (for the pretrained model).",
    )

# Define GraphState
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# System prompts
ROUTER_SYSTEM_PROMPT = """
You are an expert at routing user questions to the most appropriate information source:
1. Vectorstore: Contains documents about AI agents, prompt engineering, and adversarial attacks. Use for questions specifically on these technical AI topics.
2. Tavily Search: Use for questions about:
   - Specific people, organizations, or places
   - Recent events or current information
   - Time-sensitive queries
   - Social media profiles and current professional information
3. Wikipedia: Use for general knowledge questions about established facts, historical events, or well-documented topics.
4. Default (Pretrained Model): Use when the question:
   - Requires general reasoning
   - Doesn't fit the above categories
   - Needs creative or analytical thinking
"""

SEARCH_PERSON_PROMPT = """Based on these search results:
1. Find the exact LinkedIn profile for "{questions}"
2. Convert any 'in.linkedin.com' URLs to 'www.linkedin.com' format
3. Carefully examine the profile's contact information section and entire profile content for any GitHub links or personal websites
4. Create a professional summary based on their LinkedIn information

Search results:
{results}

Important Instructions:
- Always convert LinkedIn URLs to start with '@https://www.linkedin.com/' (not in.linkedin.com)
- Check contact information section specifically for GitHub links or personal websites
- Look for any mentions of 'github.com' or personal website URLs in the profile
- Verify the GitHub profile or website belongs to the same person

Format your response exactly as follows:

Verified Profiles:
- LinkedIn: [Must start with @https://www.linkedin.com/in/]
- GitHub: [If found in contact info or profile, must start with @https://github.com/]
- Website: [If available, must start with @http:// or @https://]

Professional Summary:
[Detailed summary including: current role, company, education, expertise areas only from the LinkedIn profile]
"""

SEARCH_GENERAL_PROMPT = """Based on these search results:
{results}

Create a concise and informative summary of the key findings related to the query "{questions}". If any relevant websites are available, list them.

Important Instructions:
- Focus on the main takeaways from the search results
- Provide URLs for relevant websites or resources
- Ensure the response is concise and well-structured

Format your response as:
Summary:
[Key findings from the search]

Websites:
- [List relevant websites here]
"""

def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store = None
        st.session_state.workflow = None
        st.session_state.llm = None
        st.session_state.retriever = None
        st.session_state.wiki_search = None
        st.session_state.tavily_client = None

def retrieve(state):
    """Vector store retrieval function"""
    print("---Retrieve---")
    questions = state['question']
    print(questions)
    docs = st.session_state.retriever.invoke(questions)
    return {'documents': docs, 'question': questions}

def wikipedia_search(state):
    """Wikipedia search function"""
    print("---Wikipedia---")
    questions = state['question']
    print(questions)
    docs = st.session_state.wiki_search.invoke(questions)
    wiki_res = Document(page_content=docs)
    return {'documents': wiki_res, 'question': questions}

def use_pretrained_model(state):
    """Default LLM function"""
    print("---Use Pretrained Model---")
    questions = state['question']
    print(questions)
    response = st.session_state.llm.invoke([HumanMessage(content=questions)])
    pretrained_res = Document(page_content=response.content)
    return {'documents': pretrained_res, 'question': questions}

def search_func(state):
    """Tavily search function"""
    print("---Tavily Search---")
    questions = state['question']
    print(questions)

    try:
        # Determine if the query is about a person
        if any(keyword in questions.lower() for keyword in ["linkedin", "github", "profile", "person"]):
            linkedin_query = f"{questions} + linkedin + github"
            linkedin_results = st.session_state.tavily_client.search(
                query=linkedin_query,
                search_depth="advanced",
                max_results=3
            )
            
            prompt = SEARCH_PERSON_PROMPT.format(questions=questions, results=linkedin_results)
            response = st.session_state.llm.invoke([HumanMessage(content=prompt)])
            
            # Post-process the response
            content = response.content
            content = content.replace("in.linkedin.com", "www.linkedin.com")
            content = content.replace("linkedin.com/in//", "linkedin.com/in/")
            
            return {'documents': Document(page_content=content), 'question': questions}
        
        else:
            general_results = st.session_state.tavily_client.search(
                query=questions,
                search_depth="basic",
                max_results=5
            )
            
            prompt = SEARCH_GENERAL_PROMPT.format(results=general_results, questions=questions)
            response = st.session_state.llm.invoke([HumanMessage(content=prompt)])
            
            return {'documents': Document(page_content=response.content), 'question': questions}
            
    except Exception as e:
        error_message = f"Error during search: {str(e)}"
        return {'documents': Document(page_content=error_message), 'question': questions}

def route_question(state):
    """Question routing function"""
    print("---Route Question---")
    questions = state['question']
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    
    structured_llm_router = st.session_state.llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({'question': questions})

    # Keywords for routing
    person_keywords = ["who", "whose", "person", "people", "professor", "student", "teacher", "researcher", "author", "profile"]
    place_keywords = ["where", "location", "place", "city", "country", "university", "college", "institute", "campus"]
    entity_keywords = ["company", "organization", "institution", "corporation", "startup", "team"]
    time_keywords = ["when", "recent", "latest", "current", "new", "update", "today", "now"]
    social_keywords = ["linkedin", "x.com", "github", "social media", "profile", "contact"]

    question_lower = questions.lower()
    
    # Check keywords
    is_person_related = any(keyword in question_lower for keyword in person_keywords)
    is_place_related = any(keyword in question_lower for keyword in place_keywords)
    is_entity_related = any(keyword in question_lower for keyword in entity_keywords)
    is_time_sensitive = any(keyword in question_lower for keyword in time_keywords)
    is_social_related = any(keyword in question_lower for keyword in social_keywords)

    if (is_person_related or is_place_related or is_entity_related or
        is_time_sensitive or is_social_related or source.datasource == 'tavily_search'):
        print("---ROUTE QUESTION TO TAVILY SEARCH---")
        return "tavily_search"
    elif source.datasource == 'wikipedia_search':
        print("---ROUTE QUESTION TO WIKI SEARCH---")
        return "wikipedia_search"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO PRETRAINED MODEL---")
        return "default"

def setup_vector_store():
    """Initialize vector store with sample documents"""
    # Sample URLs to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    # Load and process documents
    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(doc_list)
    
    # Add documents to vector store
    st.session_state.vector_store.add_documents(texts)
    st.session_state.retriever = st.session_state.vector_store.as_retriever()

def initialize_system():
    """Initialize all components"""
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    st.session_state.vector_store = Cassandra(
        embedding=embeddings,
        table_name='trial_agent_1',
        session=None,
        keyspace=None
    )
    
    # Initialize LLM
    st.session_state.llm = ChatGroq(
        groq_api_key=st.session_state.groq_api_key,
        model_name='llama-3.1-70b-Versatile'
    )
    
    # Initialize Wikipedia search
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
    st.session_state.wiki_search = WikipediaQueryRun(api_wrapper=api_wrapper)
    
    # Initialize Tavily client
    st.session_state.tavily_client = TavilyClient(api_key=st.session_state.tavily_api_key)
    
    # Setup vector store with sample documents
    setup_vector_store()
    
    # Setup workflow
    setup_workflow()

def setup_workflow():
    """Setup the workflow graph"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node('tavily_search', search_func)
    workflow.add_node('wikipedia_search', wikipedia_search)
    workflow.add_node('retrieve', retrieve)
    workflow.add_node('use_pretrained_model', use_pretrained_model)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            'tavily_search': 'tavily_search',
            'wikipedia_search': 'wikipedia_search',
            'vectorstore': 'retrieve',
            'default': 'use_pretrained_model'
        }
    )
    
    # Add edges
    workflow.add_edge('tavily_search', END)
    workflow.add_edge('retrieve', END)
    workflow.add_edge('wikipedia_search', END)
    workflow.add_edge('use_pretrained_model', END)
    
    st.session_state.workflow = workflow.compile()

def setup_api_keys():
    """Setup API keys section"""
    with st.sidebar:
        st.header("API Keys Configuration")
        astra_token = st.text_input("Astra DB Token", type="password")
        astra_db_id = st.text_input("Astra DB ID", type="password")
        groq_api_key = st.text_input("Groq API Key", type="password")
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
        if st.button("Initialize System"):
            if all([astra_token, astra_db_id, groq_api_key, tavily_api_key]):
                try:
                    # Initialize Cassandra
                    cassio.init(token=astra_token, database_id=astra_db_id)
                    
                    # Store API keys in session state
                    st.session_state.groq_api_key = groq_api_key
                    st.session_state.tavily_api_key = tavily_api_key
                    
                    # Initialize system
                    initialize_system()
                    
                    st.session_state.initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
            else:
                st.error("Please provide all API keys")

def main():
    st.title("AI Agent Query System")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup API keys section
    setup_api_keys()
    
    # Main application
    if st.session_state.initialized:
        st.header("Query System")
        
        # Input section
        query = st.text_input("Enter your question:")
        
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing query..."):
                    try:
                        # Process query
                        inputs = {'question': query}
                        results = st.session_state.workflow.stream(inputs)
                        
                        # Display results
                        for output in results:
                            for key, value in output.items():
                                st.subheader(f"Node: {key}")
                                
                                documents = value['documents']
                                if isinstance(documents, list):
                                    for doc in documents:
                                        st.write(doc.page_content)
                                else:
                                    st.write(documents.page_content)
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a query")
    else:
        st.info("Please initialize the system with your API keys in the sidebar")

if __name__ == "__main__":
    main()
