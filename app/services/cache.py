"""
app/services/cache.py
─────────────────────
Redis-backed query cache for Qdrant vector-search results.

Design decisions
────────────────
• Cache key = SHA-256(query.strip().lower())  → deterministic, compact (64 hex chars).
• Stored value = JSON-serialised list of ContextChunk dicts (same shape as
  retriever_service.search_context() output).
• TTL = 3600 s (1 hour) — long enough to benefit repeated classroom sessions,
  short enough to pick up freshly indexed Wikipedia articles.
• Graceful degradation: if Redis is unreachable every call falls back to a live
  Qdrant search.  Exceptions are logged at WARNING level, not raised.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import redis

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache TTL in seconds
CACHE_TTL = 3600

# Namespace prefix so the key space stays tidy if Redis is shared
KEY_PREFIX = "rag:ctx:"


def _make_cache_key(query: str) -> str:
    """Return a deterministic SHA-256 cache key for *query*."""
    normalised = query.strip().lower()
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def _get_redis_client() -> Optional[redis.Redis]:
    """
    Build a Redis client from settings.  Returns *None* if the connection
    cannot be established, allowing callers to degrade gracefully.
    """
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True,
            socket_connect_timeout=1,  # fail fast if Redis is down
        )
        client.ping()  # verify connectivity at creation time
        return client
    except Exception as exc:
        logger.warning(f"[Cache] Redis unavailable — caching disabled. Reason: {exc}")
        return None


# Module-level singleton; re-used across requests
_redis_client: Optional[redis.Redis] = _get_redis_client()


def get_cached_context(query: str) -> Optional[List[Dict[str, Any]]]:
    """
    Look up *query* in Redis.

    Returns a list of context-chunk dicts on a cache hit, or *None* on a miss /
    when Redis is unavailable.
    """
    if _redis_client is None:
        return None
    try:
        key = _make_cache_key(query)
        raw = _redis_client.get(key)
        if raw:
            logger.info(f"[Cache] HIT  → key={key[:20]}…")
            return json.loads(raw)
        logger.info(f"[Cache] MISS → key={key[:20]}…")
        return None
    except Exception as exc:
        logger.warning(f"[Cache] Read error: {exc}")
        return None


def set_cached_context(query: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Serialise *chunks* and store them in Redis under the SHA-256 key for
    *query*, with a TTL of :data:`CACHE_TTL` seconds.

    Silently swallows errors so callers are never disrupted by cache failures.
    """
    if _redis_client is None:
        return
    try:
        key = _make_cache_key(query)
        _redis_client.setex(key, CACHE_TTL, json.dumps(chunks))
        logger.info(f"[Cache] SET  → key={key[:20]}… (TTL={CACHE_TTL}s, chunks={len(chunks)})")
    except Exception as exc:
        logger.warning(f"[Cache] Write error: {exc}")
