import os
from itertools import chain
from operator import itemgetter
import pathlib
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from langchain_cohere import CohereEmbeddings  # Update this line
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnableLambda
from qdrant_client import models, QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from gemini import generate_context

# Constants
TOP_K = 3
MAX_DOCS_FOR_CONTEXT = 3
QDRANT_URL = (
    "https://8e0cc1a2-e1e3-40c7-af27-1ba5090d139a.us-east4-0.gcp.cloud.qdrant.io:6333"
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY", "NQrQlwkoUS-7XdScCCj7k19dSGxDm_fqejCfVCyoGahcmUA-gz_-xQ"
)  # Use your Qdrant API key
QDRANT_COLLECTION_NAME = "ALERT_DOCS"

os.environ["COHERE_API_KEY"] = (
    "GDUmx1dbg3tqdm4HV0Qg4EZIXoKKLPH5s8CVSRAT"  # Use your Cohere API key
)
embedding_model = CohereEmbeddings(
    model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY")
)
semantic_chunker_embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(
    QDRANT_URL,
    prefer_grpc=True,
    api_key="NQrQlwkoUS-7XdScCCj7k19dSGxDm_fqejCfVCyoGahcmUA-gz_-xQ",
)


def read_txt_files(directory: str) -> list[Document]:
    """Reads all .txt files in a given directory and returns a list of Document objects.

    Args:
        directory (str): The path to the directory containing .txt files.

    Returns:
        list[Document]: A list of Document objects containing the content of the .txt files.
    """
    documents = []
    for txt_file in pathlib.Path(directory).glob("*.txt"):
        with open(txt_file, "r") as file:
            content = file.read()
            metadata = {"filename": txt_file.stem}
            documents.append(Document(page_content=content, metadata=metadata))
    return documents


def reciprocal_rank_fusion(results: list[list], k=60) -> list[Document]:
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): Retrieved documents
        k (int, optional): Parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: List of documents reranked by RRF
    """
    threshold_percent = 0.75
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)  # Serialize the document to a string to use as a key
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    # Calculate the threshold based on the highest score
    max_score = max(fused_scores.values())
    score_threshold = max_score * threshold_percent

    # Filter and sort documents based on the score threshold
    reranked_results = [
        (loads(doc_str), score)
        for doc_str, score in sorted(
            fused_scores.items(), key=itemgetter(1), reverse=True
        )
        if score > score_threshold
    ]

    # Depending on the context or need, adjust MAX_DOCS_FOR_CONTEXT if necessary
    return [doc for doc, _ in reranked_results[:MAX_DOCS_FOR_CONTEXT]]


def query_generator(original_query: dict) -> list[str]:
    """Generate queries from original query

    Args:
        original_query (dict): Original query

    Returns:
        list[str]: List of generated queries
    """
    query = original_query.get("query")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates multiple search queries based on a single input query.",
            ),
            (
                "user",
                "Generate multiple search queries related to: {original_query}. When creating queries, please refine or add closely related contextual information, without significantly altering the original query's meaning.",
            ),
            ("user", "OUTPUT (3 queries):"),
        ]
    )

    model = Cohere()
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    queries = query_generator_chain.invoke({"original_query": query})
    queries.insert(0, "0. " + query)

    return queries


def create_QDrant_collection():
    """Create Qdrant collection."""
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )


def upload_chunks_to_QDrant(documents):
    records_to_upload = []
    for idx, chunk in enumerate(documents):
        content = chunk.page_content
        # Change how you call to get the embedding
        vector = embedding_model.embed_documents([content])[
            0
        ]  # Check if this is the correct call

        record = models.PointStruct(
            id=idx, vector=vector, payload={"page_content": content}
        )
        records_to_upload.append(record)

    qdrant_client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME, points=records_to_upload
    )
    return


def alert_retriever(query: str) -> list[Document]:
    """RRF retriever

    Args:
        query (str): Query string
        directory (str): Directory containing the .txt files

    Returns:
        list[Document]: Retrieved documents
    """
    # Read documents from the directory

    # Initialize Qdrant vector store
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding_model,
    )

    # Set up the retriever with search kwargs
    retriever = qdrant.as_retriever(search_kwargs={"k": TOP_K})

    # RRF chain setup
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()  # Use the Qdrant retriever
        | reciprocal_rank_fusion
    )

    # Invoke the chain with the query
    result = chain.invoke({"query": query})

    # Print the top K matched documents
    print("Top K Matched Documents:")
    for idx, document in enumerate(result[:TOP_K]):
        print(
            f"Document {idx + 1}:\nFilename: {document.metadata.get('filename', 'N/A')}, Content: {document.page_content[:100]}...\n"
        )

    return result


def classify_alert():
    alert_text = """
    Alert Name:
    Unauthorized Access Attempt

    Description:
    Multiple failed login attempts detected from IP address 192.168.1.45. Immediate review required.
    """
    return generate_context(alert_retriever(alert_text), alert_text)


"""
if __name__ == "__main__":


    
    
    # create_QDrant_collection()
    
    directory_list = ['Spyware', 'Trojan', 'Worm', 'Fileless Malware'] 
    # ['Adware', 'Backdoor', 'Cryptominer', 'Polymorphic Malware', 'Ransomware', 'Rootkit', 'Spyware', 'Trojan', 'Worm', 'Fileless Malware' ]
    # Example usage
    final_docs=[]
    for directory_name in directory_list:
        directory = f'datasets/{directory_name}'  # Directory containing the .txt files
        documents = read_txt_files(directory)
        final_docs.extend(documents)
    print(final_docs)
    upload_chunks_to_QDrant(final_docs)
"""
