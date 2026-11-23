"""Configure the embedding model used for job descriptions and queries."""

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


def create_embeddings() -> HuggingFaceEmbeddings:
    """Load BGE-M3 on CPU and return an encoder for jobs and queries."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        # Encode up to eight texts at a time and scale each vector to unit length.
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )
