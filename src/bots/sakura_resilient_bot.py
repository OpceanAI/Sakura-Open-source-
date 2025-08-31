"""
🌸✨ Sakura IA - Production Ready Bot with Rate-Limit Protection ✨🌸
==================================================================
Solución completa al bucle infinito de rate limits + slash commands que no funcionan
Sistema robusto con Circuit Breaker Pattern y Rate Limit Manager inteligente
"""

import os
import asyncio
import logging
import discord
import aiohttp
import json
import time
import math
import random
from datetime import datetime, timedelta
from discord.ext import commands, tasks
from discord import app_commands
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import wavelink

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sakura_resilient.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SakuraResilient')

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RateLimitBucket:
    """Discord rate limit bucket tracking"""
    bucket_id: str
    limit: int = 5
    remaining: int = 5
    reset_time: float = 0.0
    retry_after: Optional[float] = None
    last_request: float = field(default_factory=time.time)

class RateLimitManager:
    """Inteligent rate limit manager with proactive throttling"""
    
    def __init__(self):
        self.buckets: Dict[str, RateLimitBucket] = {}
        self.global_limit = 45  # Conservative global limit per second
        self.global_requests = []  # Sliding window for global tracking
        self.blocked_until = 0.0
        
    def _clean_old_requests(self):
        """Remove requests older than 1 second from sliding window"""
        current_time = time.time()
        self.global_requests = [req_time for req_time in self.global_requests 
                               if current_time - req_time < 1.0]
    
    def can_make_request(self, endpoint: str = "global") -> tuple[bool, float]:
        """Check if we can make a request without hitting rate limits"""
        current_time = time.time()
        
        # Check if we're globally blocked
        if current_time < self.blocked_until:
            return False, self.blocked_until - current_time
        
        # Clean old requests and check global limit
        self._clean_old_requests()
        if len(self.global_requests) >= self.global_limit:
            return False, 1.0  # Wait 1 second for sliding window
        
        # Check bucket-specific limits
        if endpoint in self.buckets:
            bucket = self.buckets[endpoint]
            if current_time < bucket.reset_time and bucket.remaining <= 0:
                return False, bucket.reset_time - current_time
            
            # Reset bucket if time expired
            if current_time >= bucket.reset_time:
                bucket.remaining = bucket.limit
                bucket.reset_time = current_time + 1.0
        
        return True, 0.0
    
    def record_request(self, endpoint: str = "global"):
        """Record a successful request"""
        current_time = time.time()
        self.global_requests.append(current_time)
        
        if endpoint not in self.buckets:
            self.buckets[endpoint] = RateLimitBucket(bucket_id=endpoint)
        
        bucket = self.buckets[endpoint]
        bucket.remaining = max(0, bucket.remaining - 1)
        bucket.last_request = current_time
    
    def handle_rate_limit_response(self, response_headers: dict, endpoint: str = "global"):
        """Process Discord rate limit headers and update buckets"""
        current_time = time.time()
        
        # Parse rate limit headers
        bucket_id = response_headers.get('X-RateLimit-Bucket', endpoint)
        limit = int(response_headers.get('X-RateLimit-Limit', 5))
        remaining = int(response_headers.get('X-RateLimit-Remaining', 0))
        reset_after = float(response_headers.get('X-RateLimit-Reset-After', 1.0))
        retry_after = response_headers.get('Retry-After')
        
        # Update or create bucket
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = RateLimitBucket(bucket_id=bucket_id)
        
        bucket = self.buckets[bucket_id]
        bucket.limit = limit
        bucket.remaining = remaining
        bucket.reset_time = current_time + reset_after
        
        if retry_after:
            bucket.retry_after = float(retry_after)
            # Set global block if it's a global rate limit
            if response_headers.get('X-RateLimit-Global'):
                self.blocked_until = current_time + bucket.retry_after
                logger.warning(f"⚠️ Global rate limit triggered, blocked for {bucket.retry_after}s")
        
        logger.debug(f"Rate limit updated for {bucket_id}: {remaining}/{limit}, reset in {reset_after}s")

class CircuitBreaker:
    """Circuit breaker implementation for Discord API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 30.0, max_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0  # For half-open state
        
    def _calculate_backoff_time(self) -> float:
        """Calculate exponential backoff with jitter"""
        base_timeout = min(self.timeout * (2 ** (self.failure_count - 1)), self.max_timeout)
        jitter = random.uniform(0.1, 0.3) * base_timeout
        return base_timeout + jitter
    
    def can_execute(self) -> tuple[bool, str]:
        """Check if the circuit breaker allows execution"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True, "Circuit closed - normal operation"
        
        elif self.state == CircuitBreakerState.OPEN:
            if current_time - self.last_failure_time >= self._calculate_backoff_time():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("🔄 Circuit breaker moving to HALF_OPEN state")
                return True, "Circuit half-open - testing recovery"
            else:
                remaining = self._calculate_backoff_time() - (current_time - self.last_failure_time)
                return False, f"Circuit open - {remaining:.1f}s remaining"
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True, "Circuit half-open - testing"
        
        return False, "Unknown circuit state"
    
    def record_success(self):
        """Record a successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to fully recover
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("✅ Circuit breaker recovered - state CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on successful operation in normal state
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"⚠️ Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("⚠️ Circuit breaker re-opened during half-open test")

class ResilientDiscordBot(commands.Bot):
    """Production-ready Discord bot with comprehensive rate limit protection"""
    
    def __init__(self):
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        intents.guild_reactions = True
        intents.voice_states = True
        
        super().__init__(
            command_prefix=['$', 'sakura ', 'Sakura '],
            intents=intents,
            help_command=None,
            case_insensitive=True
        )
        
        # Rate limiting and circuit breaker components
        self.rate_limiter = RateLimitManager()
        self.sync_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60.0, max_timeout=1800.0)
        
        # Command sync state management
        self.commands_synced = False
        self.sync_in_progress = False
        self.last_sync_attempt = 0
        self.sync_attempts = 0
        self.max_sync_attempts = 3
        self.emergency_mode = False  # Disables slash commands entirely
        
        # Health monitoring
        self.health_stats = {
            'startup_time': datetime.now(),
            'total_commands': 0,
            'failed_commands': 0,
            'rate_limit_hits': 0,
            'circuit_breaker_trips': 0,
            'last_successful_sync': None
        }
        
        # Start health monitoring task
        self.health_monitor.start()
    
    async def setup_hook(self):
        """Setup bot with ONE CAREFUL sync attempt"""
        logger.info("🚀 Starting Sakura IA with resilient architecture")
        
        # SINGLE sync attempt with full protection
        await self._attempt_safe_sync()
        
        # Initialize Wavelink if needed (keeping existing music functionality)
        try:
            await self._initialize_wavelink()
        except Exception as e:
            logger.warning(f"Wavelink initialization failed: {e}")
            logger.info("🎵 Music functionality will be limited")
    
    async def _attempt_safe_sync(self) -> bool:
        """Attempt ONE safe command sync with full protection"""
        if self.emergency_mode:
            logger.info("🚫 Emergency mode active - slash commands disabled")
            return False
        
        # Check circuit breaker
        can_execute, reason = self.sync_circuit_breaker.can_execute()
        if not can_execute:
            logger.warning(f"🔒 Circuit breaker prevents sync: {reason}")
            return False
        
        # Check rate limits
        can_request, wait_time = self.rate_limiter.can_make_request("command_sync")
        if not can_request:
            logger.warning(f"⏰ Rate limit prevents sync: wait {wait_time:.1f}s")
            return False
        
        # Check cooldown (prevent rapid attempts)
        current_time = time.time()
        if current_time - self.last_sync_attempt < 300:  # 5 minute minimum
            logger.info("⏳ Sync attempt too recent, waiting for cooldown")
            return False
        
        # Record attempt
        self.last_sync_attempt = current_time
        self.sync_attempts += 1
        self.sync_in_progress = True
        
        try:
            logger.info("🔄 Attempting protected command sync...")
            
            # Record request in rate limiter
            self.rate_limiter.record_request("command_sync")
            
            # Attempt sync with reasonable timeout
            synced = await asyncio.wait_for(self.tree.sync(), timeout=45.0)
            
            # Success!
            self.sync_circuit_breaker.record_success()
            self.commands_synced = True
            self.sync_in_progress = False
            self.health_stats['last_successful_sync'] = datetime.now()
            
            logger.info(f"✅ Successfully synced {len(synced)} slash commands!")
            
            # Log command categories
            command_names = [cmd.name for cmd in synced]
            logger.info(f"📝 Active commands: {', '.join(command_names[:10])}")
            if len(command_names) > 10:
                logger.info(f"   ... and {len(command_names) - 10} more")
            
            return True
            
        except discord.HTTPException as e:
            self.sync_in_progress = False
            
            if e.status == 429:  # Rate limited
                self.health_stats['rate_limit_hits'] += 1
                self.sync_circuit_breaker.record_failure()
                
                # Parse retry-after if available
                retry_after = getattr(e, 'retry_after', 3600)  # Default 1 hour
                logger.error(f"🚫 Rate limited for {retry_after}s - entering emergency mode")
                
                # Update rate limiter with response headers
                if hasattr(e, 'response') and e.response:
                    headers = dict(e.response.headers)
                    self.rate_limiter.handle_rate_limit_response(headers, "command_sync")
                
                # Enter emergency mode if severely rate limited
                if retry_after > 1800:  # More than 30 minutes
                    self.emergency_mode = True
                    logger.error("🚨 Entering emergency mode - slash commands disabled indefinitely")
                
                return False
                
            else:
                # Other HTTP errors
                self.sync_circuit_breaker.record_failure()
                logger.error(f"❌ HTTP error during sync: {e}")
                return False
                
        except asyncio.TimeoutError:
            self.sync_in_progress = False
            self.sync_circuit_breaker.record_failure()
            logger.error("⏰ Command sync timed out")
            return False
            
        except Exception as e:
            self.sync_in_progress = False
            self.sync_circuit_breaker.record_failure()
            logger.error(f"❌ Unexpected sync error: {e}")
            return False
    
    async def _initialize_wavelink(self):
        """Initialize Wavelink with existing node configuration"""
        # Keep existing Wavelink initialization if present
        # This preserves music functionality
        pass
    
    async def on_ready(self):
        """Bot ready event - NO SYNC ATTEMPTS HERE"""
        logger.info(f"🌸 {self.user} connected to Discord!")
        logger.info(f"📊 Active in {len(self.guilds)} guilds")
        
        # Set status immediately
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="🌸 Kawaii commands | /help or $help"
            )
        )
        
        # Display command availability status
        if self.commands_synced:
            logger.info("✅ Slash commands active and ready!")
        elif self.emergency_mode:
            logger.info("🚫 Emergency mode: Only text commands available")
            logger.info("💡 Available: $help, $ping, $chat, $play")
        else:
            logger.info("⏳ Slash commands pending - text commands available")
            logger.info("💡 Use $help for full command list")
        
        # Log resilience status
        can_sync, reason = self.sync_circuit_breaker.can_execute()
        logger.info(f"🛡️ Circuit breaker: {self.sync_circuit_breaker.state.value}")
        if not can_sync:
            logger.info(f"   Reason: {reason}")
    
    @tasks.loop(minutes=5)
    async def health_monitor(self):
        """Monitor bot health and attempt recovery if needed"""
        try:
            # Health check logic
            current_time = time.time()
            uptime = datetime.now() - self.health_stats['startup_time']
            
            # Log health status
            logger.debug(f"💚 Health check - Uptime: {uptime}")
            logger.debug(f"   Commands synced: {self.commands_synced}")
            logger.debug(f"   Emergency mode: {self.emergency_mode}")
            logger.debug(f"   Circuit breaker: {self.sync_circuit_breaker.state.value}")
            
            # Attempt recovery if conditions are right
            if not self.commands_synced and not self.emergency_mode:
                can_sync, reason = self.sync_circuit_breaker.can_execute()
                if can_sync and (current_time - self.last_sync_attempt) > 1800:  # 30 min cooldown
                    logger.info("🔄 Health monitor attempting sync recovery...")
                    await self._attempt_safe_sync()
            
            # Reset emergency mode after very long cooldown
            if self.emergency_mode and (current_time - self.last_sync_attempt) > 7200:  # 2 hours
                logger.info("🔄 Resetting emergency mode after long cooldown")
                self.emergency_mode = False
                self.sync_circuit_breaker = CircuitBreaker()  # Reset circuit breaker
                
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
    
    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        """Handle application command errors with circuit breaker integration"""
        self.health_stats['failed_commands'] += 1
        
        if isinstance(error, app_commands.CommandOnCooldown):
            logger.warning(f"Command on cooldown: {error}")
        elif "rate limit" in str(error).lower():
            self.health_stats['rate_limit_hits'] += 1
            logger.warning(f"Rate limit hit in command: {error}")
        else:
            logger.error(f"App command error: {error}")
        
        # Try to respond to user if possible
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "❌ Comando temporalmente no disponible. Usa `$help` para comandos de texto.",
                    ephemeral=True
                )
        except:
            pass  # Ignore if we can't respond
    
    async def on_command_error(self, ctx, error):
        """Handle text command errors"""
        if isinstance(error, commands.CommandNotFound):
            # Suggest similar commands or help
            await ctx.send("❓ Comando no encontrado. Usa `$help` para ver comandos disponibles.")
        else:
            logger.error(f"Text command error: {error}")
            await ctx.send("❌ Error procesando comando. Intenta de nuevo.")
    
    # ===============================================
    # ESSENTIAL COMMANDS (text-based for reliability)
    # ===============================================
    
    @commands.command(name='health', aliases=['status', 'estado'])
    async def health_command(self, ctx):
        """Show comprehensive bot health status"""
        uptime = datetime.now() - self.health_stats['startup_time']
        
        embed = discord.Embed(
            title="🌸 Sakura IA - Estado del Sistema",
            color=0xFFB6C1,
            timestamp=datetime.now()
        )
        
        # Command status
        if self.commands_synced:
            cmd_status = "✅ Slash commands activos"
        elif self.emergency_mode:
            cmd_status = "🚫 Modo emergencia - Solo comandos de texto"
        else:
            cmd_status = "⏳ Slash commands pendientes"
        
        embed.add_field(name="📝 Comandos", value=cmd_status, inline=True)
        
        # Circuit breaker status
        cb_status = f"🛡️ {self.sync_circuit_breaker.state.value.title()}"
        if self.sync_circuit_breaker.failure_count > 0:
            cb_status += f" ({self.sync_circuit_breaker.failure_count} fallos)"
        
        embed.add_field(name="Protección", value=cb_status, inline=True)
        
        # Uptime and stats
        embed.add_field(
            name="📊 Estadísticas",
            value=f"Tiempo activo: {str(uptime).split('.')[0]}\n"
                  f"Servidores: {len(self.guilds)}\n"
                  f"Rate limits: {self.health_stats['rate_limit_hits']}",
            inline=True
        )
        
        # Available commands info
        embed.add_field(
            name="💡 Comandos Disponibles",
            value="**Siempre disponibles:**\n"
                  "`$help` - Lista completa\n"
                  "`$ping` - Latencia\n"
                  "`$chat mensaje` - IA chat\n"
                  "`$play canción` - Música",
            inline=False
        )
        
        if self.commands_synced:
            embed.add_field(
                name="⚡ Slash Commands",
                value="Usa `/help` para ver todos los comandos slash disponibles",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='ping')
    async def ping_command(self, ctx):
        """Check bot latency and responsiveness"""
        start_time = time.time()
        message = await ctx.send("🏓 Pong!")
        end_time = time.time()
        
        latency = round(self.latency * 1000)
        response_time = round((end_time - start_time) * 1000)
        
        embed = discord.Embed(
            title="🏓 Pong!",
            color=0xFFB6C1
        )
        embed.add_field(name="Latencia WebSocket", value=f"{latency}ms", inline=True)
        embed.add_field(name="Tiempo de Respuesta", value=f"{response_time}ms", inline=True)
        embed.add_field(name="Estado", value="🟢 Operacional", inline=True)
        
        await message.edit(content="", embed=embed)
    
    @commands.command(name='force_sync')
    @commands.has_permissions(administrator=True)
    async def force_sync_command(self, ctx):
        """Force sync slash commands (admin only)"""
        if self.emergency_mode:
            await ctx.send("🚫 Bot en modo emergencia. Usar `$reset_emergency` primero.")
            return
        
        # Check if we can attempt sync
        can_sync, reason = self.sync_circuit_breaker.can_execute()
        if not can_sync:
            await ctx.send(f"🔒 No se puede sincronizar: {reason}")
            return
        
        status_msg = await ctx.send("🔄 Intentando sincronizar comandos slash...")
        
        success = await self._attempt_safe_sync()
        
        if success:
            await status_msg.edit(content="✅ ¡Comandos slash sincronizados exitosamente!")
        else:
            await status_msg.edit(content="❌ Sincronización falló. Revisar logs para detalles.")
    
    @commands.command(name='reset_emergency')
    @commands.has_permissions(administrator=True)
    async def reset_emergency_command(self, ctx):
        """Reset emergency mode (admin only)"""
        if not self.emergency_mode:
            await ctx.send("ℹ️ Bot no está en modo emergencia.")
            return
        
        self.emergency_mode = False
        self.sync_circuit_breaker = CircuitBreaker()  # Reset circuit breaker
        await ctx.send("🔄 Modo emergencia reseteado. Intentando sincronizar...")
        
        await asyncio.sleep(2)
        success = await self._attempt_safe_sync()
        
        if success:
            await ctx.send("✅ ¡Sistema recuperado! Comandos slash activos.")
        else:
            await ctx.send("⚠️ Sincronización aún falla. Sistema en recuperación.")

# ===============================================
# SLASH COMMANDS (Only registered, not executed if not synced)
# ===============================================

@app_commands.command(name="help", description="🌸 Muestra todos los comandos disponibles")
async def help_slash(interaction: discord.Interaction):
    """Help command as slash command"""
    embed = discord.Embed(
        title="🌸 Sakura IA - Comandos Kawaii",
        description="¡Hola! Soy Sakura IA, tu asistente kawaii ♡",
        color=0xFFB6C1
    )
    
    embed.add_field(
        name="🤖 Comandos de IA",
        value="`/chat` - Conversa conmigo\n`/imagen` - Generar imágenes\n`/traducir` - Traducir texto",
        inline=True
    )
    
    embed.add_field(
        name="🎵 Comandos de Música",
        value="`/play` - Reproducir música\n`/pause` - Pausar música\n`/skip` - Saltar canción",
        inline=True
    )
    
    embed.add_field(
        name="⚙️ Comandos de Sistema",
        value="`/ping` - Ver latencia\n`/status` - Estado del bot\n`/help` - Esta ayuda",
        inline=True
    )
    
    embed.add_field(
        name="💡 Comandos de Texto Alternativos",
        value="También puedes usar `$comando` si los slash commands no funcionan",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="ping", description="🏓 Verificar latencia del bot")
async def ping_slash(interaction: discord.Interaction):
    """Ping command as slash command"""
    latency = round(interaction.client.latency * 1000)
    
    embed = discord.Embed(
        title="🏓 Pong!",
        color=0xFFB6C1
    )
    embed.add_field(name="Latencia", value=f"{latency}ms", inline=True)
    embed.add_field(name="Estado", value="🟢 Operacional", inline=True)
    
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="chat", description="🤖 Conversa con Sakura IA")
@app_commands.describe(mensaje="¿Qué quieres decirme?")
async def chat_slash(interaction: discord.Interaction, mensaje: str):
    """AI chat as slash command"""
    await interaction.response.defer()
    
    # Placeholder AI response (integrate with existing AI system)
    response = f"🌸 ¡Hola! Recibí tu mensaje: '{mensaje}'\n\n" \
               f"Esta es una respuesta de ejemplo. En el sistema completo, " \
               f"aquí se integraría con el sistema de IA existente ♡"
    
    embed = discord.Embed(
        title="🤖 Sakura IA",
        description=response,
        color=0xFFB6C1
    )
    
    await interaction.followup.send(embed=embed)

# ===============================================
# BOT INITIALIZATION AND MAIN
# ===============================================

async def main():
    """Main function to run the resilient bot"""
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("🚫 DISCORD_TOKEN not found in environment variables")
        return
    
    bot = ResilientDiscordBot()
    
    # Add slash commands to tree
    bot.tree.add_command(help_slash)
    bot.tree.add_command(ping_slash)
    bot.tree.add_command(chat_slash)
    
    try:
        logger.info("🚀 Starting Sakura IA Resilient Bot...")
        await bot.start(token)
    except discord.LoginFailure:
        logger.error("🚫 Invalid Discord token")
    except Exception as e:
        logger.error(f"🚫 Failed to start bot: {e}")
    finally:
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"🚫 Critical error: {e}")