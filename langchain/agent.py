from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from plot_tools import create_pie_chart

load_dotenv()

tavily_search = TavilySearchResults(max_results=3)

tools = [tavily_search, create_pie_chart]

llm = ChatOpenAI(model="gpt-4o")

memory = SqliteSaver.from_conn_string(":memory:")

config = {"configurable": {"thread_id": "abc12"}}

agent_executor = create_react_agent(llm, tools=tools, checkpointer=memory)


while True:
    question = input()
    if question == "exit":
        break
    else:
        result = agent_executor.invoke(
            {"messages": [("human", f"{question}")]}, config=config
        )
        print(result["messages"][-1].content.replace(".", "!"))
