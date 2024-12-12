from functools import wraps
from typing import Optional
import asyncio
import json
from datetime import timedelta
import aioredis

class Cache:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)
    
    async def set(self, key: str, value: str, expire: int = 3600):
        await self.redis.set(key, value, ex=expire)
    
    def cached(self, expire: int = 3600):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key = f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}"
                
                # Try to get from cache
                cached_value = await self.get(key)
                if cached_value:
                    return json.loads(cached_value)
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(key, json.dumps(result), expire)
                return result
            return wrapper
        return decorator 