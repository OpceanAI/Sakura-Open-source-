import redis
import json
import asyncio
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import os
from functools import wraps

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Advanced Redis Cache Manager for Discord Bot
    Handles caching, rate limiting, session management, and real-time data
    """
    
    def __init__(self):
        self.redis_client = None
        self.connected = False
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.username = os.getenv('REDIS_USERNAME', 'default')
        self.password = os.getenv('REDIS_PASSWORD')
        
    async def connect(self):
        """Initialize Redis connection with retry logic"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await asyncio.to_thread(self.redis_client.ping)
            self.connected = True
            logger.info("✅ Redis Cloud connection established")
            
            # Initialize default keys
            await self._initialize_default_data()
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.connected = False
    
    async def _initialize_default_data(self):
        """Initialize default cache structure"""
        try:
            # Bot statistics
            if not await asyncio.to_thread(self.redis_client.exists, "bot:stats"):
                default_stats = {
                    "commands_executed": 0,
                    "users_served": 0,
                    "servers_active": 0,
                    "uptime_start": datetime.utcnow().isoformat()
                }
                await self.set_json("bot:stats", default_stats)
            
            logger.info("🔧 Redis default data initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis data: {e}")
    
    # === BASIC CACHE OPERATIONS ===
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set string value with TTL"""
        if not self.connected:
            return False
        try:
            return await asyncio.to_thread(self.redis_client.setex, key, ttl, value)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get string value"""
        if not self.connected:
            return None
        try:
            return await asyncio.to_thread(self.redis_client.get, key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set_json(self, key: str, data: Dict, ttl: int = 3600) -> bool:
        """Set JSON data with TTL"""
        try:
            json_data = json.dumps(data, default=str)
            return await self.set(key, json_data, ttl)
        except Exception as e:
            logger.error(f"Redis SET_JSON error: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Dict]:
        """Get JSON data"""
        try:
            data = await self.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis GET_JSON error: {e}")
            return None
    
    # === AI CONVERSATION CACHE ===
    
    async def cache_ai_response(self, user_id: str, prompt: str, response: str, model: str = "unknown"):
        """Cache AI responses for faster repeated queries"""
        cache_key = f"ai:response:{hash(prompt) % 1000000}"
        cache_data = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "usage_count": 1
        }
        await self.set_json(cache_key, cache_data, ttl=1800)  # 30 minutes
    
    async def get_cached_ai_response(self, prompt: str) -> Optional[str]:
        """Get cached AI response if available"""
        cache_key = f"ai:response:{hash(prompt) % 1000000}"
        cached = await self.get_json(cache_key)
        if cached:
            # Update usage count
            cached["usage_count"] = cached.get("usage_count", 0) + 1
            await self.set_json(cache_key, cached, ttl=1800)
            return cached["response"]
        return None
    
    # === USER SESSION MANAGEMENT ===
    
    async def set_user_session(self, user_id: str, session_data: Dict, ttl: int = 7200):
        """Set user session data (2 hours default)"""
        session_key = f"session:user:{user_id}"
        session_data["last_activity"] = datetime.utcnow().isoformat()
        await self.set_json(session_key, session_data, ttl)
    
    async def get_user_session(self, user_id: str) -> Optional[Dict]:
        """Get user session data"""
        session_key = f"session:user:{user_id}"
        return await self.get_json(session_key)
    
    async def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        session = await self.get_user_session(user_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            await self.set_user_session(user_id, session)
    
    # === RATE LIMITING ===
    
    async def check_rate_limit(self, user_id: str, command: str, limit: int = 5, window: int = 60) -> bool:
        """Check if user is rate limited for a command"""
        rate_key = f"rate:{user_id}:{command}"
        try:
            current = await asyncio.to_thread(self.redis_client.get, rate_key)
            if current is None:
                await asyncio.to_thread(self.redis_client.setex, rate_key, window, 1)
                return True
            
            if int(current) >= limit:
                return False
            
            await asyncio.to_thread(self.redis_client.incr, rate_key)
            return True
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    async def get_rate_limit_info(self, user_id: str, command: str) -> Dict:
        """Get rate limit information for user/command"""
        rate_key = f"rate:{user_id}:{command}"
        try:
            current = await asyncio.to_thread(self.redis_client.get, rate_key)
            ttl = await asyncio.to_thread(self.redis_client.ttl, rate_key)
            return {
                "current_usage": int(current) if current else 0,
                "reset_in": ttl if ttl > 0 else 0
            }
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {"current_usage": 0, "reset_in": 0}
    
    # === MUSIC QUEUE CACHE ===
    
    async def cache_music_queue(self, guild_id: str, queue: List[Dict]):
        """Cache music queue for persistence"""
        queue_key = f"music:queue:{guild_id}"
        await self.set_json(queue_key, {"queue": queue, "updated": datetime.utcnow().isoformat()}, ttl=3600)
    
    async def get_music_queue(self, guild_id: str) -> Optional[List[Dict]]:
        """Get cached music queue"""
        queue_key = f"music:queue:{guild_id}"
        cached = await self.get_json(queue_key)
        return cached.get("queue", []) if cached else None
    
    async def cache_current_track(self, guild_id: str, track_info: Dict):
        """Cache currently playing track"""
        track_key = f"music:current:{guild_id}"
        await self.set_json(track_key, track_info, ttl=600)  # 10 minutes
    
    # === BOT STATISTICS ===
    
    async def increment_command_count(self, command_name: str = "unknown"):
        """Increment command usage statistics"""
        try:
            # Global command count
            await asyncio.to_thread(self.redis_client.incr, "stats:commands:total")
            
            # Per-command count
            await asyncio.to_thread(self.redis_client.incr, f"stats:commands:{command_name}")
            
            # Daily stats
            today = datetime.utcnow().strftime("%Y-%m-%d")
            await asyncio.to_thread(self.redis_client.incr, f"stats:daily:{today}")
            
        except Exception as e:
            logger.error(f"Stats increment error: {e}")
    
    async def get_bot_statistics(self) -> Dict:
        """Get comprehensive bot statistics"""
        try:
            stats = {
                "total_commands": await asyncio.to_thread(self.redis_client.get, "stats:commands:total") or "0",
                "daily_commands": {},
                "popular_commands": {},
                "active_sessions": 0
            }
            
            # Get daily stats for last 7 days
            for i in range(7):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_count = await asyncio.to_thread(self.redis_client.get, f"stats:daily:{date}")
                stats["daily_commands"][date] = int(daily_count) if daily_count else 0
            
            # Get popular commands
            command_keys = await asyncio.to_thread(self.redis_client.keys, "stats:commands:*")
            for key in command_keys[:10]:  # Top 10
                if key != "stats:commands:total":
                    command_name = key.split(":")[-1]
                    count = await asyncio.to_thread(self.redis_client.get, key)
                    stats["popular_commands"][command_name] = int(count) if count else 0
            
            # Count active sessions
            session_keys = await asyncio.to_thread(self.redis_client.keys, "session:user:*")
            stats["active_sessions"] = len(session_keys)
            
            return stats
        except Exception as e:
            logger.error(f"Stats retrieval error: {e}")
            return {}
    
    # === GUILD SETTINGS CACHE ===
    
    async def cache_guild_settings(self, guild_id: str, settings: Dict):
        """Cache guild-specific settings"""
        settings_key = f"guild:settings:{guild_id}"
        await self.set_json(settings_key, settings, ttl=7200)  # 2 hours
    
    async def get_guild_settings(self, guild_id: str) -> Optional[Dict]:
        """Get cached guild settings"""
        settings_key = f"guild:settings:{guild_id}"
        return await self.get_json(settings_key)
    
    # === UTILITY METHODS ===
    
    async def clear_user_cache(self, user_id: str):
        """Clear all cache for a specific user"""
        try:
            pattern = f"*{user_id}*"
            keys = await asyncio.to_thread(self.redis_client.keys, pattern)
            if keys:
                await asyncio.to_thread(self.redis_client.delete, *keys)
            logger.info(f"🧹 Cleared cache for user {user_id}")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    async def get_cache_info(self) -> Dict:
        """Get Redis cache information"""
        try:
            info = await asyncio.to_thread(self.redis_client.info)
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "Unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected": self.connected
            }
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {"connected": False, "error": str(e)}
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            await asyncio.to_thread(self.redis_client.ping)
            return True
        except Exception:
            self.connected = False
            return False

# Decorator for caching function results
def redis_cache(ttl: int = 300):
    """Decorator to cache function results in Redis"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"func:{func.__name__}:{hash(str(args) + str(kwargs)) % 1000000}"
            
            # Try to get from cache first
            if hasattr(wrapper, '_redis_manager'):
                cached_result = await wrapper._redis_manager.get(cache_key)
                if cached_result:
                    try:
                        return json.loads(cached_result)
                    except:
                        pass
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            if hasattr(wrapper, '_redis_manager') and result is not None:
                try:
                    await wrapper._redis_manager.set(cache_key, json.dumps(result, default=str), ttl)
                except:
                    pass
            
            return result
        return wrapper
    return decorator

# Global Redis manager instance
redis_manager = RedisManager()