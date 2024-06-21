from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [("system", "Translate the following to {language}"), ("human", "{text}")]
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"language": "spanish", "text": "Glückunsch! Das war richtig!"})

print(result)
