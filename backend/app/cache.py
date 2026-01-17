from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple

try:
    from config import get_settings
except ImportError:
    from config import get_settings


class AsyncCache:
    """Simple async cache abstraction.

    - Uses Redis when REDIS_URL is provided.
    - Falls back to in-memory dict for dev.
    - Values are JSON-serialized for consistency across backends.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = None
        self._mem: Dict[str, Tuple[str, Optional[float]]] = {}

    async def initialize(self) -> None:
        if self._settings.redis_url:
            from redis import asyncio as aioredis  # lazy import

            try:
                self._redis = aioredis.from_url(self._settings.redis_url, decode_responses=True)
                # Validate connectivity; fallback to memory on failure
                await self._redis.ping()
            except Exception:
                self._redis = None

    async def get(self, key: str) -> Optional[Any]:
        if self._redis is not None:
            value = await self._redis.get(key)
            if value is None:
                return None
            return json.loads(value)

        # in-memory
        now = time.time()
        packed = self._mem.get(key)
        if not packed:
            return None
        value_str, expires_at = packed
        if expires_at is not None and now >= expires_at:
            # expired
            self._mem.pop(key, None)
            return None
        return json.loads(value_str)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        value_str = json.dumps(value)
        if self._redis is not None:
            if ttl_seconds is None:
                await self._redis.set(key, value_str)
            else:
                await self._redis.set(key, value_str, ex=ttl_seconds)
            return

        expires_at = None if ttl_seconds is None else time.time() + ttl_seconds
        self._mem[key] = (value_str, expires_at)

    async def ping(self) -> bool:
        if self._redis is not None:
            try:
                await self._redis.ping()
                return True
            except Exception:
                return False
        return True

    def backend_name(self) -> str:
        return "redis" if self._settings.redis_url and self._redis is not None else "memory"


