REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"
EMBEDDER_PARAMS = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",  ##nomic-embed-text-v1
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": False},
}
TOP_K = 10
