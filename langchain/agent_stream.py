import asyncio
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver

from plot_tools import create_pie_chart, create_bar_chart, create_line_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

search = TavilySearchResults(max_results=3)

tools = [search, create_pie_chart, create_bar_chart, create_line_graph]

memory = MemorySaver()

config = {"configurable": {"thread_id": "abc12"}}

agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)


async def stream_output(question):
    async for event in agent_executor.astream_events(
        {"messages": [("human", f"{question}")]}, config=config, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="")


async def main():
    while True:
        print("\n")
        question = input("Eingabe: ")
        if question == "exit":
            break
        else:
            await stream_output(question)


if __name__ == "__main__":
    asyncio.run(main())
