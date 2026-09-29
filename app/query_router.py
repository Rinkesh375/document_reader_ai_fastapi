from dotenv import load_dotenv
from fastapi import APIRouter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

openai_client = OpenAI() 


router = APIRouter(
    prefix="/query",
    tags=["query"],
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333",
)


@router.post("/")
async def query(question: str):
    search_results = vector_db.similarity_search(query=question)

    print(f"Search results for question: {question}: {search_results}")
    
    context = " ".join(
    [
        f"[Page {result.metadata.get('page', 'unknown')}] {result.page_content}"
        for result in search_results
    ]
    )
    
    print(f"Context:{context}")

    SYSTEM_PROMPT = f"""
    You are a helpful assistant that answers questions based on the provided context.
    If the context does not contain the answer, respond with "I don't know."
    {context}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":question},
        ],
        max_completion_tokens=1000
    )

    return {
        "question": question,
        "results": response.choices[0].message.content
    }
