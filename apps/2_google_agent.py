from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # <-- Updated to a supported Llama 3.1 version
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY") 
)
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model = llm,
    tools = [search.run],
    checkpointer= memory,
    system_prompt = "You are a agent and can search for any question on google "
)

while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("GoodBye")
        break
    
    response = agent.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}},
        )
    print("AI:",response["messages"][-1].content) 