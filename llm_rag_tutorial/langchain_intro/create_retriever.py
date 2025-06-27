import dotenv
from langchain_community.document_loaders import CSVLoader

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from constants import REVIEWS_CSV_PATH, REVIEWS_CHROMA_PATH, EMBEDDER_PARAMS

## Load embeddings (Probably sentence bert embeddings/ nomic embeddings etc)

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# model_name = "sentence-transformers/all-mpnet-base-v2"  ##nomic-embed-text-v1
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": False}  ## Already normalized
embedding = HuggingFaceEmbeddings(**EMBEDDER_PARAMS)

reviews_chroma_db = Chroma.from_documents(
    documents=reviews,
    embedding=embedding,
    persist_directory=REVIEWS_CHROMA_PATH,
)

# if __name__ == "__main__":
#     loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
#     reviews = loader.load()
#     for i in range(10):
#         print(reviews[i].page_content[:100])
#         print(reviews[i].metadata)
#         print("***************************")
