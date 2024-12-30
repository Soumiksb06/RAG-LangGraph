# Streamlit App for LangGraph AI Agents
import streamlit as st
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from pprint import pprint

# Define helper functions
def retrieve(state):
    st.write("---Retrieve---")
    questions = state['question']
    st.write(questions)
    docus = retriever.invoke(questions)
    return {'documents': docus, 'question': questions}

def wikipedia_search(state):
    st.write("---Wikipedia Search---")
    questions = state['question']
    st.write(questions)
    docus = wiki_search.run(questions)
    wiki_res = Document(page_content=docus)
    return {'documents': wiki_res, 'question': questions}

def use_pretrained_model(state):
    st.write("---Use Pretrained Model---")
    questions = state['question']
    st.write(questions)
    response = llm.invoke([HumanMessage(content=questions)])
    pretrained_res = Document(page_content=response.content)
    return {'documents': pretrained_res, 'question': questions}

def search_func(state):
    st.write("---Tavily Search---")
    questions = state['question']
    st.write(questions)
    try:
        linkedin_query = f"{questions} + linkedin + github"
        linkedin_results = tavily_client.search(
            query=linkedin_query,
            search_depth="advanced",
            max_results=3
        )
        prompt = f"""Based on these search results:
        {linkedin_results}

        Verified Profiles:
        - LinkedIn: [Must start with @https://www.linkedin.com/in/]
        - GitHub: [If found in contact info or profile, must start with @https://github.com/]

        Professional Summary:
        [Detailed summary from the LinkedIn profile]
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("in.linkedin.com", "www.linkedin.com")
        return {'documents': Document(page_content=content), 'question': questions}
    except Exception as e:
        error_message = f"Error during search: {str(e)}"
        return {'documents': Document(page_content=error_message), 'question': questions}

def route_question(state):
    st.write("---Route Question---")
    questions = state['question']
    source = question_router.invoke({'question': questions})
    question_lower = questions.lower()
    if source.datasource == 'tavily_search':
        return "tavily_search"
    elif source.datasource == 'wikipedia_search':
        return "wikipedia_search"
    elif source.datasource == 'vectorstore':
        return "retrieve"
    else:
        return "default"

# Workflow setup
workflow = StateGraph(GraphState)
workflow.add_node('tavily_search', search_func)
workflow.add_node('wikipedia_search', wikipedia_search)
workflow.add_node('retrieve', retrieve)
workflow.add_node('use_pretrained_model', use_pretrained_model)
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
workflow.add_edge('tavily_search', END)
workflow.add_edge('retrieve', END)
workflow.add_edge('wikipedia_search', END)
workflow.add_edge('use_pretrained_model', END)
app = workflow.compile()

# Streamlit interface
st.title("LangGraph AI Agents")
st.write("Ask a question and let the app route it to the appropriate system.")

# User input
user_question = st.text_input("Enter your question:", placeholder="E.g., What is an agent?")
if st.button("Submit"):
    inputs = {'question': user_question}
    for output in app.stream(inputs):
        for key, value in output.items():
            st.write(f"Node '{key}':")
            documents = value['documents']
            if isinstance(documents, list):
                for doc in documents:
                    st.write(doc.page_content)
            else:
                st.write(documents.page_content)
