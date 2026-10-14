import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIKI_COLLECTION_NAME = "wikipedia_chunks"

EMBEDDING_VECTOR_SIZE = 384

def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    return client

def init_qdrant_db():
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        existing_names = [col.name for col in collections]

        if WIKI_COLLECTION_NAME not in existing_names:
            logger.info(f"Creating collection '{WIKI_COLLECTION_NAME}' in Qdrant...")
            client.create_collection(
                collection_name=WIKI_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    # Pin vectors in RAM — eliminates disk I/O on every ANN search
                    on_disk=False,
                ),
                # HNSW graph config: m=16 is a good default for 384-dim embeddings.
                # ef_construct=100 → higher graph quality at index-build time.
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10_000,
                ),
                # Keep optimizer threshold generous so small batches don't force merges
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                ),
            )
            logger.info(f"Collection '{WIKI_COLLECTION_NAME}' successfully created!")
        else:
            logger.info(f"Collection '{WIKI_COLLECTION_NAME}' already exists in Qdrant.")
    except Exception as e:
            logger.error(f"Failed to initialize Qdrant database: {e}")
            raise e

qdrant_client = get_qdrant_client()
