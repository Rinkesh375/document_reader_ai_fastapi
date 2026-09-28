from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "data" / "sample.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=400
)

chunk = text_splitter.split_documents(documents=docs)


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunk,
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
)


print("Vector store created and documents embedded successfully.")