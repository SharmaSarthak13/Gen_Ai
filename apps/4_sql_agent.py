import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 1. Database Setup (Runs on script load/rerun safely)
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# 2. System Prompt
system_prompt = """
You are a task management assistant that interacts with a SQLite database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to a maximum of 10 results, always ordered by 'created_at' DESC unless the user explicitly requests otherwise.
2. After executing a CREATE, UPDATE, or DELETE operation, you MUST run a follow-up SELECT query to confirm the changes were successful before responding.
3. If the user requests a list or view of tasks, always present the final output clearly in a markdown table format.
4. CRITICAL SAFETY: Never execute destructive commands like DROP TABLE. Never execute a DELETE or UPDATE query without a specific WHERE clause.

DATABASE SCHEMA REFERENCE:
Table Name: tasks
Columns:
- id: INTEGER PRIMARY KEY AUTOINCREMENT
- title: TEXT NOT NULL
- description: TEXT
- status: TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending'
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

# 3. Cached Agent Factory (CRITICAL FIX: Everything inside the cache)
@st.cache_resource
def get_agent():
    # Initialize the model inside the cache
    model = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY") 
    )
    
    # Generate tools inside the cache
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    cached_tools = toolkit.get_tools()
    
    # Initialize memory inside the cache
    memory = MemorySaver()
    
    # Create the agent with stably bound tools
    return create_react_agent(
        model=model,
        tools=cached_tools,
        checkpointer=memory,
        prompt=system_prompt
    )

agent = get_agent()

# 4. Streamlit UI
st.subheader("TaskBot - Manage your todos")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User Input
prompt = st.chat_input("Ask me to manage your tasks?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("ai"):
        with st.spinner("Processing..."):
            # Execute agent via LangGraph react agent schema
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "1"}},
            )
            result = response["messages"][-1].content
            st.markdown(result)    
            st.session_state.messages.append({"role": "ai", "content": result})