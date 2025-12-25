# from __future__ import annotations

# from typing import Protocol, Sequence

# # Define a generic interface for vector store backends
# class VectorBackend(Protocol):
#     def index(self, texts: list[str]) -> None: ...
#     def add_text(self, text: str, vector_id: int | str | None = None, metadata: dict | None = None) -> None: ...
#     def search(
#         self, query: str, k: int = 5,
#         since: str | None = None,
#         min_importance: float = 0.0,
#         type_filter: str | None = None,
#     ) -> list[str]: ...

# # Qdrant vector store integration
# HEAVY_QDRANT = False
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http.models import (
#         Distance, VectorParams, PointStruct, Filter,
#         FieldCondition, Range, MatchValue,
#     )
# except Exception:
#     QdrantClient = None # type: ignore

# except ImportError as e:
#     # If Qdrant client is not installed, raise an error with guidance
#     raise ImportError("Qdrant client not installed. Please install 'qdrant-client' to use QdrantMemoryBackend.") from e

# from loguru import logger

# if HAVE_QDRANT:
#     class QdrantMemoryBackend:
#         """
#         A VectorBackend implementation using Qdrant as the vector database.
#         Handles connection to a local or remote Qdrant instance and provides methods
#         to add vectors and query them for semantic similarity.
#         """
#         def __init__(self, embedder, dim: int, *, url: str | None = None,
#                     api_key: str | None = None, collection_name: str = "events"):
#             """
#             Initialize the Qdrant vector store backend.
#             - embedder: an object with an .encode(texts: Sequence[str]) -> np.ndarray method for embedding generation.
#             - dim: dimensionality of the embedding vectors.
#             - url/api_key: if provided, connect to a remote Qdrant instance (e.g. Qdrant Cloud URL and API key).
#             If not provided, connect to a local Qdrant server at localhost.
#             - collection_name: name of the Qdrant collection to use for storing vectors.
#             """
#             self.embedder = embedder
#             self.dim = dim
#             self.collection_name = collection_name
#             # Connect to Qdrant (remote or local)
#             if url:
#                 logger.info(f"QdrantMemoryBackend: Connecting to remote Qdrant at {url}")
#                 self.client = QdrantClient(url=url, api_key=api_key)  # Use HTTPS URL and optional API key
#             else:
#                 logger.info("QdrantMemoryBackend: Connecting to local Qdrant (localhost:6333)")
#                 self.client = QdrantClient(host="localhost", port=6333)
#             # Ensure the collection exists with correct vector size and distance
#             try:
#                 # If collection does not exist, create it
#                 self.client.get_collection(collection_name=self.collection_name)
#                 logger.debug(f"Collection '{self.collection_name}' already exists in Qdrant.")
#             except Exception:
#                 # Create collection with specified vector parameters (use Cosine for semantic similarity)
#                 logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={dim}, metric=Cosine)")
#                 self.client.recreate_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
#                 )

#         def index(self, texts: list[str]) -> None:
#             """
#             Index a batch of texts into Qdrant.
#             Each text will be embedded and inserted as a new vector point.
#             Note: If possible, provide unique IDs for each text via add_event to ensure traceability.
#             """
#             if not texts or self.embedder is None:
#                 return
#             # Generate embeddings for all texts (in batch)
#             try:
#                 vectors = self.embedder.encode(texts)
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.index: Embedding batch failed: {e}")
#                 return
#             # Prepare PointStruct list for Qdrant upsert
#             points = []
#             for i, vec in enumerate(vectors):
#                 # Use a generated ID (timestamp-based) if no explicit ID is provided.
#                 point_id = None
#                 try:
#                     # Create a unique ID using current time in milliseconds plus index
#                     import time
#                     point_id = int(time.time() * 1000) + i
#                 except Exception:
#                     point_id = None  # Let Qdrant auto-assign if time is not available
#                 payload = {"content": texts[i]}
#                 points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
#             # Upsert points into Qdrant collection
#             try:
#                 self.client.upsert(collection_name=self.collection_name, points=points)
#                 logger.info(f"QdrantMemoryBackend: Indexed {len(points)} new vectors into collection '{self.collection_name}'.")
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.index: Failed to upsert points to Qdrant: {e}")

#         def add_text(self, text: str, vector_id: int | str | None = None, metadata: dict | None = None) -> None:
#             """
#             Add a single text with a specified ID and metadata to the Qdrant collection.
#             This ensures traceability by using a known ID (e.g., the event ID from the MemoryStore).
#             """
#             if self.embedder is None:
#                 return
#             payload = metadata.copy() if metadata else {}
#             payload.setdefault("content", text)
#             # Generate embedding vector for the text
#             try:
#                 vec = self.embedder.encode([text])[0]
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.add_text: Embedding failed for ID {vector_id}: {e}")
#                 return
#             point = PointStruct(id=vector_id if vector_id is not None else 0,  # use 0 or auto-ID if None
#                                 vector=vec.tolist(), payload=payload)
#             try:
#                 self.client.upsert(collection_name=self.collection_name, points=[point])
#                 logger.debug(f"QdrantMemoryBackend: Inserted vector ID {point.id} into '{self.collection_name}'.")
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.add_text: Qdrant upsert failed for ID {vector_id}: {e}")

#         def search(self, query: str, k: int = 5,
#                 since: str | None = None,
#                 min_importance: float = 0.0,
#                 type_filter: str | None = None) -> list[str]:
#             """
#             Perform a semantic search for the given query text in the vector store.
#             Returns up to k most similar event contents, optionally filtered by time, importance, or type.
#             - since: If provided, only consider events with timestamp >= since (ISO 8601 string).
#             - min_importance: If > 0, only consider events with importance >= that value.
#             - type_filter: If provided, only consider events of this type.
#             """
#             if self.embedder is None:
#                 return []
#             # Embed the query text to a vector
#             try:
#                 query_vec = self.embedder.encode([query])[0]
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.search: Query embedding failed: {e}")
#                 return []
#             # Build Qdrant filter conditions based on parameters
#             qdrant_filter: Filter | None = None
#             conditions = []
#             if since is not None:
#                 # Filter for events with timestamp >= since (compare numeric epoch if available, or ISO string lexicographically)
#                 try:
#                     # If since is ISO string, convert to epoch seconds for numeric comparison if possible
#                     import datetime
#                     dt = datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))
#                     since_epoch = int(dt.timestamp())
#                     conditions.append(FieldCondition(key="timestamp", range=Range(gte=since_epoch)))
#                 except Exception:
#                     # Fallback: treat since as string directly
#                     conditions.append(FieldCondition(key="ts_iso", range=Range(gte=since)))
#             if min_importance is not None and min_importance > 0:
#                 conditions.append(FieldCondition(key="importance", range=Range(gte=min_importance)))
#             if type_filter:
#                 conditions.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))
#             if conditions:
#                 qdrant_filter = Filter(must=conditions)
#             # Perform vector similarity search in Qdrant
#             try:
#                 results = self.client.search(
#                     collection_name=self.collection_name,
#                     query_vector=query_vec.tolist(),
#                     limit=k,
#                     with_payload=True,
#                     filter=qdrant_filter
#                 )
#             except Exception as e:
#                 logger.error(f"QdrantMemoryBackend.search: Qdrant search failed: {e}")
#                 return []
#             if not results:
#                 return []
#             # Combine similarity score with importance for ranking (score + normalized importance)
#             scored_points = list(results)
#             for pt in scored_points:
#                 imp = 0.0
#                 if pt.payload and "importance" in pt.payload:
#                     # importance might be stored as float or int in payload
#                     try:
#                         imp = float(pt.payload.get("importance", 0.0))
#                     except Exception:
#                         imp = 0.0
#                 # Augment point with a combined score attribute (for sorting)
#                 pt.combined_score = pt.score + (imp / 100.0)
#             # Sort by combined score descending
#             scored_points.sort(key=lambda p: getattr(p, "combined_score", p.score), reverse=True)
#             # Extract content field from payload of top results
#             top_contents: list[str] = []
#             for pt in scored_points[:k]:
#                 if pt.payload and "content" in pt.payload:
#                     top_contents.append(pt.payload["content"])
#                 else:
#                     # If content not stored, skip or attempt to retrieve by ID (not implemented here)
#                     logger.warning(f"QdrantMemoryBackend.search: No content payload for result ID {pt.id}")
#             logger.debug(f"QdrantMemoryBackend.search: Query='{query}' -> {len(top_contents)} results")
#             return top_contents
