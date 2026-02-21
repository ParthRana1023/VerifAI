"""
Cache service using ChromaDB for semantic caching of scraped web articles.

Uses sentence-transformers (all-MiniLM-L6-v2) for local embeddings — no API keys needed.
Data persists to ./chroma_cache/ directory.
"""

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

# Persistent storage directory
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_cache")
COLLECTION_NAME = "scraped_articles"
CACHE_TTL_DAYS = 7
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score to consider a match (0-1 scale)

# Lazy-loaded globals
_client = None
_collection = None


def _get_embedding_function():
    """Get the sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def _get_collection():
    """Get or create the ChromaDB collection (lazy-loaded singleton)."""
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"description": "Cached scraped news articles for VerifAI"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents)",
            COLLECTION_NAME,
            _collection.count(),
        )
        return _collection
    except Exception as e:
        logger.error("Failed to initialize ChromaDB: %s", e)
        return None


def _url_hash(url: str) -> str:
    """Create a deterministic ID from a URL for deduplication."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def search_cache(query: str, n_results: int = 5) -> list[dict]:
    """
    Semantically search the cache for articles matching the query.

    Returns a list of dicts with keys: title, url, source, content_snippet, similarity, cached_at
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []

    # Clean expired articles first
    _clear_expired()

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        articles = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns L2 distance; convert to similarity (lower = more similar)
                # Normalize: similarity = 1 / (1 + distance)
                distance = results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)

                if similarity < SIMILARITY_THRESHOLD:
                    continue

                meta = results["metadatas"][0][i]
                articles.append({
                    "title": meta.get("title", "Unknown"),
                    "url": meta.get("url", ""),
                    "source": meta.get("source", "Unknown"),
                    "content_snippet": (results["documents"][0][i] or "")[:300],
                    "similarity": round(similarity, 3),
                    "cached_at": meta.get("cached_at", "Unknown"),
                })

        logger.info("Cache search for '%s': found %d relevant articles", query, len(articles))
        return articles

    except Exception as e:
        logger.error("Error searching cache: %s", e)
        return []


def cache_articles(query: str, articles: list[dict]) -> int:
    """
    Cache a list of articles in ChromaDB.

    Each article dict should have: title, url, source, and optionally content_snippet.
    Returns the number of newly cached articles.
    """
    collection = _get_collection()
    if collection is None:
        return 0

    cached_count = 0
    now = datetime.now(timezone.utc).isoformat()

    for article in articles:
        url = article.get("url", "")
        if not url:
            continue

        doc_id = _url_hash(url)

        # Skip if already cached (dedup by URL)
        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            continue

        # Build the document text for embedding
        title = article.get("title", "Unknown")
        source = article.get("source", "Unknown")
        snippet = article.get("content_snippet", "")
        document_text = f"{title}. {snippet}" if snippet else title

        try:
            collection.add(
                ids=[doc_id],
                documents=[document_text],
                metadatas=[{
                    "title": title,
                    "url": url,
                    "source": source,
                    "query": query,
                    "cached_at": now,
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(days=CACHE_TTL_DAYS)
                    ).isoformat(),
                }],
            )
            cached_count += 1
        except Exception as e:
            logger.warning("Failed to cache article '%s': %s", title, e)

    logger.info("Cached %d new articles for query '%s'", cached_count, query)
    return cached_count


def _clear_expired():
    """Remove articles older than CACHE_TTL_DAYS."""
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return

    try:
        # Fetch all documents to check expiry (ChromaDB doesn't support TTL natively)
        all_docs = collection.get(include=["metadatas"])
        if not all_docs or not all_docs["ids"]:
            return

        now = datetime.now(timezone.utc)
        expired_ids = []

        for i, doc_id in enumerate(all_docs["ids"]):
            expires_at = all_docs["metadatas"][i].get("expires_at", "")
            if expires_at:
                try:
                    expiry = datetime.fromisoformat(expires_at)
                    if now > expiry:
                        expired_ids.append(doc_id)
                except ValueError:
                    pass

        if expired_ids:
            collection.delete(ids=expired_ids)
            logger.info("Cleared %d expired cached articles", len(expired_ids))

    except Exception as e:
        logger.warning("Error clearing expired cache: %s", e)


def get_cache_stats() -> dict:
    """Get basic stats about the cache."""
    collection = _get_collection()
    if collection is None:
        return {"status": "unavailable", "count": 0}
    return {"status": "active", "count": collection.count()}
