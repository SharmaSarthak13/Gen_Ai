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

# Initialize session state keys cleanly
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []
    
# Setup agent
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing AI agent and can search on Google"
)

#### Building Web Interface code 
st.subheader("QuickAnswer - Answers at the speed of thought")

# Render existing chat history
for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

# Capture user input
query = st.chat_input("Ask Anything ?")

if query:
    # 1. Display and save user message
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    
    # 2. Setup the stream output container for the AI
    with st.chat_message("assistant"):
        space = st.empty()
        message_accumulator = ""
        
        # 3. Request the stream from the LangGraph agent
        response_stream = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="messages"
        )
        
        # 4. Correctly unpack the (message_chunk, metadata) tuple
        for message_chunk, metadata in response_stream:
            # Check if the chunk contains text content to avoid errors
            if message_chunk.content:
                message_accumulator += message_chunk.content
                space.markdown(message_accumulator) # Use markdown to render cleanly
                
        # 5. Save the final completed message to the session state history
        if message_accumulator:
            st.session_state.history.append({"role": "assistant", "content": message_accumulator})