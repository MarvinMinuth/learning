from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

embedding = OpenAIEmbeddings()


def embedding_function():
    return embedding


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embedding, persist_directory="vectorstore"
)
