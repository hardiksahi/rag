EMBEDDER_PARAMS = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",  ##nomic-embed-text-v1
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": False},
}
REVIEW_TOP_K = 10
CYPHER_TOP_K = 100
