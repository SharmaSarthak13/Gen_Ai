import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    streaming=True,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY") 
)

# Setup search tool
search = GoogleSerperAPIWrapper()
tools = [search.run] 

# FIX 1 & 2: Initialize session state keys cleanly without using 'query'
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    # Start with a clean, empty chat history list
    st.session_state.history = []
    
# Setup agent
agent = create_agent(
    model=llm,
    tools=tools,
    streaming = True,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing AI agent and can search on Google"
)

#### Building Web Interface code 
st.subheader("QuickAnswer - Answers at the speed of thought")

# Render existing chat history
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# Capture user input
query = st.chat_input("Ask Anything ?")

if query:
    # 1. Display and save user message
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    
    # 2. Get response from the LangGraph agent
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode = "messages"
    )
    
    ai_container = st.chat_messages("Ai")
    with ai_container:
        space = st.empty()
        
        message = ""
        
        for chunk in response:
             message = message + chunk[0].content
             space.write(message) 
             
        st.session_state.history.append({"role": "assistant", "content": message})