from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from simple_embeddings import embedding_function


load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

vectorstore = Chroma(
    persist_directory="vectorstore", embedding_function=embedding_function()
)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is TreeSize?")
