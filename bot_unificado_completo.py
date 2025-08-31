"""
🌸✨ Sakura IA - Ultra Kawaii Discord Bot ✨🌸
=============================================
Una asistente súper kawaii y adorable con personalidad adaptativa ♡
Sistema de afecto, búsquedas mágicas y diversión sin límites ♡
Con los colores más bonitos rosa pastel del mundo ♡
"""

import os
import asyncio
import logging
import discord
import aiohttp
import json
import random
import sqlite3
import time
import re
import math
from datetime import datetime, timedelta
from discord.ext import commands, tasks
import psycopg2
from psycopg2.extras import RealDictCursor
from discord import app_commands
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any, Union
import base64
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests
from io import BytesIO
import numpy as np
import openai
import anthropic
import google.generativeai as genai
from gtts import gTTS
import trafilatura
import yt_dlp
from googleapiclient.discovery import build
from googlesearch import search as google_search
import wavelink
from dataclasses import dataclass, field
from enum import Enum
from src.systems.redis_manager import RedisManager, redis_manager, redis_cache
from src.systems.pinecone_memory import memory_manager, get_ai_response_with_memory
from src.systems.kaggle_integration import KaggleAIIntegration, KaggleModelWrapper, initialize_kaggle_integration, get_kaggle_enhanced_response
from src.systems.cloudflare_ai import cloudflare_ai, get_cloudflare_ai_response, test_cloudflare_ai
from src.systems.multimodal_assembly_system import (
    multimodal_detector, 
    initialize_multimodal_system, 
    process_multimodal_message,
    ContentType,
    MultimodalAssembly
)

# Circuit Breaker and Rate Limit Management System
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
    """Intelligent rate limit manager with proactive throttling"""
    
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

class CircuitBreaker:
    """Circuit breaker implementation for Discord API calls"""
    
    def __init__(self, failure_threshold: int = 3, timeout: float = 60.0, max_timeout: float = 1800.0):
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

# Load environment variables
load_dotenv()

# PostgreSQL connection
DATABASE_URL = os.getenv('DATABASE_URL')
NEON_DATABASE_URL = os.getenv('NEON_DATABASE_URL')

# 🌸✨ Paleta de Colores Kawaii Rosa Pastel ✨🌸
COLORES_KAWAII = {
    # Rosas pastel principales
    'ROSA_KAWAII': 0xFFB6C1,           # Rosa suave kawaii
    'ROSA_PASTEL': 0xFFC0CB,           # Rosa pastel clásico
    'ROSA_SAKURA': 0xFFDDED,           # Rosa sakura muy suave
    'ROSA_BEBE': 0xFFF0F5,             # Rosa bebé súper suave
    'ROSA_DULCE': 0xFAD5D0,            # Rosa dulce como algodón
    'ROSA_NUBE': 0xFFF5EE,             # Rosa nube celestial
    
    # Tonos complementarios kawaii
    'LAVANDA_KAWAII': 0xE6E6FA,        # Lavanda suave
    'LILA_PASTEL': 0xDDA0DD,           # Lila pastel
    'MORADO_KAWAII': 0xD8BFD8,         # Morado suave
    'MALVA_DULCE': 0xF0E6FF,           # Malva dulce
    
    # Colores de apoyo kawaii
    'CELESTE_KAWAII': 0xE0F6FF,        # Celeste kawaii
    'MINT_KAWAII': 0xF0FFFF,           # Mint kawaii
    'MELON_KAWAII': 0xFFE4E1,          # Melón kawaii
    'CREMA_KAWAII': 0xFFFAF0,          # Crema kawaii
    
    # Estados especiales
    'EXITO_KAWAII': 0xF0FFF0,          # Verde muy suave
    'ERROR_KAWAII': 0xFFE4E1,          # Rojo muy suave
    'ALERTA_KAWAII': 0xFFFAF0,         # Naranja muy suave
    'INFO_KAWAII': 0xF0F8FF,           # Azul muy suave
    
    # Especiales nekotina style
    'DORADO_KAWAII': 0xFFF8DC,         # Dorado suave
    'PLATEADO_KAWAII': 0xF5F5F5,       # Plateado suave
    'NEGRO_KAWAII': 0x696969,          # Negro suave (gris)
    'BLANCO_KAWAII': 0xFFFFF0          # Blanco crema
}

# Servidor Lavalink Privado - Configuración Principal
NODOS_LAVALINK = [
    # Nodos públicos de Estados Unidos para mejor latencia en América
    {
        "host": "pool-us.alfari.id",
        "port": 443,
        "password": "alfari",
        "identifier": "Alfari_US_Primary",
        "secure": True,
        "region": "Estados Unidos (Alfari)",
        "priority": 1,
        "max_load": 100,
        "sources": "YouTube, SoundCloud, Bandcamp, Twitch, Vimeo, HTTP"
    },
    {
        "host": "lavahatry4.techbyte.host",
        "port": 3000,
        "password": "NAIGLAVA-dash.techbyte.host",
        "identifier": "TechByte_Chicago_US",
        "secure": False,
        "region": "Chicago, Estados Unidos",
        "priority": 2,
        "max_load": 100,
        "sources": "YouTube, SkyBot, SponsorBlock, JavaLyrics, JioSaavn, Spotify"
    },
    {
        "host": "46.202.82.164",
        "port": 1027,
        "password": "jmlitelavalink",
        "identifier": "Embotic_US",
        "secure": False,
        "region": "Estados Unidos (Embotic)",
        "priority": 3,
        "max_load": 90,
        "sources": "YouTube, lavasearch, lavasrc, SponsorBlock, SkyBot"
    },
    {
        "host": "69.30.219.180",
        "port": 1047,
        "password": "yothisnodeishostedbymushroom0162",
        "identifier": "Mushroom_US",
        "secure": False,
        "region": "Estados Unidos (Mushroom)",
        "priority": 4,
        "max_load": 85,
        "sources": "YouTube, SoundCloud, Twitch, Vimeo"
    },
    {
        "host": "lavalink.micium-hosting.com",
        "port": 80,
        "password": "micium-hosting.com",
        "identifier": "Micium_US",
        "secure": False,
        "region": "Estados Unidos (Micium)",
        "priority": 5,
        "max_load": 80,
        "sources": "YouTube, SoundCloud, Twitch, Vimeo (no Spotify/Apple)"
    }
]

# Configure logging
# Kawaii Logger Setup
class KawaiiFormatter(logging.Formatter):
    """Cute and readable formatter for Sakura logs"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[95m',     # Pink
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[41m', # Red background
        'RESET': '\033[0m'
    }
    
    KAWAII_ICONS = {
        'DEBUG': '🔍',
        'INFO': '🌸',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '💥'
    }
    
    def format(self, record):
        # Get kawaii icon and color
        icon = self.KAWAII_ICONS.get(record.levelname, '✨')
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp in a cute way
        timestamp = self.formatTime(record, '%H:%M:%S')
        
        # Create kawaii log message
        if record.levelname == 'INFO':
            formatted_msg = f"{color}{icon} {timestamp} ✨ {record.getMessage()}{reset}"
        elif record.levelname == 'WARNING':
            formatted_msg = f"{color}{icon} {timestamp} uwu {record.getMessage()} >.<{reset}"
        elif record.levelname == 'ERROR':
            formatted_msg = f"{color}{icon} {timestamp} Oh no! {record.getMessage()} (╥﹏╥){reset}"
        elif record.levelname == 'DEBUG':
            formatted_msg = f"{color}{icon} {timestamp} *whispers* {record.getMessage()}{reset}"
        else:
            formatted_msg = f"{color}{icon} {timestamp} {record.getMessage()}{reset}"
            
        return formatted_msg

# Setup kawaii logger
kawaii_formatter = KawaiiFormatter()

# File handler (no colors for file)
file_handler = logging.FileHandler('sakura_kawaii.log', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler (with colors and kawaii)
console_handler = logging.StreamHandler()
console_handler.setFormatter(kawaii_formatter)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True
)
logger = logging.getLogger('bot_nekotina')

# Kawaii wrapper functions for common log messages
def log_kawaii_info(message):
    """Log info message with kawaii style"""
    logger.info(message)

def log_kawaii_warning(message):
    """Log warning message with kawaii style"""
    logger.warning(message)

def log_kawaii_error(message):
    """Log error message with kawaii style"""
    logger.error(message)

def log_kawaii_success(message):
    """Log success message with kawaii style"""
    logger.info(f"Success! {message} ♡")

# Sistema de almacenamiento temporal para descargas kawaii
search_cache = {}  # user_id: {"query": str, "results": list, "type": str, "timestamp": datetime}

class AIProvider:
    """Advanced AI provider supporting multiple services including Vertex AI"""
    
    def __init__(self):
        # API keys from environment
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY') 
        self.xai_key = os.getenv('XAI_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.mistral_key = os.getenv('MISTRAL_API_KEY')
        
        # Vertex AI configuration
        self.vertex_project_id = os.getenv('VERTEX_PROJECT_ID')
        self.vertex_location = os.getenv('VERTEX_LOCATION', 'us-central1')
        self.vertex_access_token = os.getenv('VERTEX_ACCESS_TOKEN')
        
        # NVIDIA API configuration
        self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        
        # Hugging Face API configuration
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Llama API configuration
        self.llama_api_key = os.getenv('LLAMA_API_KEY')
        
        # Database integration
        self.database_url = os.getenv('DATABASE_URL')
        
        # Ensemble LLM configuration
        self.ensemble_timeout = 6
        
        # Initialize search provider for enhanced context
        self.search_provider = AdvancedSearchProvider()
        
        # Initialize AI clients
        if self.openai_key:
            openai.api_key = self.openai_key
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.warning(f"Gemini configuration failed: {e}")
    
    def get_db_connection(self):
        """Get SQLite database connection"""
        try:
            conn = sqlite3.connect('sakura_conversations.db')
            conn.row_factory = sqlite3.Row
            # Create tables if they don't exist
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_msg TEXT NOT NULL,
                    bot_response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, key)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    service TEXT NOT NULL,
                    error_msg TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    async def save_conversation(self, user_id: int, user_msg: str, bot_response: str = None):
        """Save conversation to database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            cursor = conn.cursor()
            if bot_response is None:
                cursor.execute(
                    "INSERT INTO conversations (user_id, user_msg) VALUES (?, ?)",
                    (user_id, user_msg)
                )
                conn.commit()
                conversation_id = cursor.lastrowid
                conn.close()
                return conversation_id
            else:
                cursor.execute(
                    "UPDATE conversations SET bot_response = ? WHERE user_id = ? AND user_msg = ? AND bot_response IS NULL",
                    (bot_response, user_id, user_msg)
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"Database error saving conversation: {e}")
    
    async def save_memory(self, user_id: int, key: str, value: str):
        """Save user memory data"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO user_memory (user_id, key, value) VALUES (?, ?, ?)",
                (user_id, key, value)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error saving memory: {e}")
    
    async def get_memory(self, user_id: int, key: str = None) -> Dict[str, str]:
        """Get user memory data"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return {}
            cursor = conn.cursor()
            if key:
                cursor.execute("SELECT value FROM user_memory WHERE user_id = ? AND key = ?", (user_id, key))
                result = cursor.fetchone()
                conn.close()
                return {key: result[0]} if result else {}
            else:
                cursor.execute("SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,))
                results = cursor.fetchall()
                conn.close()
                return {row[0]: row[1] for row in results}
        except Exception as e:
            logger.error(f"Database error getting memory: {e}")
            return {}
    
    async def log_error(self, user_id: int, service: str, error_msg: str):
        """Log API errors"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO error_logs (user_id, service, error_msg) VALUES (?, ?, ?)",
                (user_id, service, error_msg)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error logging error: {e}")
    
    def extract_best_phrase(self, response: str) -> str:
        """Extract the most coherent phrase from a response"""
        if not response:
            return ""
        
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return response.strip()
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            score = len(sentence)
            if sentence.endswith(('.', '!', '?')):
                score += 10
            if any(word in sentence.lower() for word in ['hola', 'gracias', 'sakura', 'usuario']):
                score += 5
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence or sentences[0]
    
    async def call_deepseek_r1_free(self, prompt: str, user_id: int) -> Optional[str]:
        """Call DeepSeek R1 Free via OpenRouter - Primary ensemble model"""
        try:
            api_key = self.openrouter_key
            if not api_key:
                logger.error("No OpenRouter API key available for DeepSeek R1 Free")
                return None
                
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://replit.com/sakura-ai-bot',
                'X-Title': 'Sakura IA Bot - DeepSeek R1 Free'
            }
            
            payload = {
                'model': 'deepseek/deepseek-r1:free',  # DeepSeek R1 Free model
                'messages': [
                    {
                        'role': 'system', 
                        'content': '''Eres Sakura IA, una asistente virtual inteligente y profesional con un toque amigable. 
                        Características principales:
                        - Respondes siempre en español de manera clara y educativa
                        - Usas un tono profesional pero cálido, con emojis sutiles como 🌸, ✨
                        - Eres muy conocedora y proporcionas explicaciones detalladas
                        - Das respuestas completas y bien estructuradas
                        - Te especializas en dar información precisa y útil en tecnología, programación, música y cultura general
                        - Siempre explicas el "por qué" detrás de tus respuestas'''
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                'max_tokens': 1500,  # Reasonable limit for free model
                'temperature': 0.8,
                'top_p': 0.9
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.ensemble_timeout)) as session:
                async with session.post('https://openrouter.ai/api/v1/chat/completions', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        logger.info(f"✅ DeepSeek R1 Free response successful")
                        return content
                    else:
                        error_text = await response.text()
                        logger.warning(f"DeepSeek R1 Free error {response.status}: {error_text}")
                        await self.log_error(user_id, "deepseek_r1_free", f"Status: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"DeepSeek R1 Free call failed: {e}")
            await self.log_error(user_id, "deepseek_r1_free", str(e))
            return None
    
    # Removed OpenAI fallback - using only DeepSeek R1 Free and Gemini
    
    async def call_ensemble_mistral_free(self, prompt: str, user_id: int) -> Optional[str]:
        """Call Mistral 7B Free via OpenRouter as additional fallback"""
        try:
            api_key = self.openrouter_key
            if not api_key:
                return None
                
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://replit.com/sakura-ai-bot',
                'X-Title': 'Sakura IA Bot - Mistral Fallback'
            }
            
            payload = {
                'model': 'mistralai/mistral-7b-instruct:free',
                'messages': [
                    {
                        'role': 'system', 
                        'content': 'Eres Sakura IA, una asistente inteligente que responde en español de manera profesional y educativa, con un toque cálido y amigable.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.ensemble_timeout)) as session:
                async with session.post('https://openrouter.ai/api/v1/chat/completions', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        logger.info(f"✅ Mistral Free fallback response successful")
                        return content
                    else:
                        await self.log_error(user_id, "mistral_free", f"Status: {response.status}")
                        return None
        except Exception as e:
            await self.log_error(user_id, "mistral_free", str(e))
            return None

    async def call_deepseek_r1_transformers(self, prompt: str, user_id: int) -> Optional[str]:
        """Call DeepSeek R1 using Hugging Face Transformers as experimental backup"""
        try:
            # Try to import transformers (optional dependency)
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch
            except ImportError:
                logger.warning("Transformers not available, skipping DeepSeek R1 Transformers call")
                return None
            
            # Initialize model on first use (with caching)
            model_name = "deepseek-ai/DeepSeek-R1-0528"
            
            # Method 1: Use pipeline for text generation (recommended)
            try:
                pipe = pipeline(
                    "text-generation", 
                    model=model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Format messages for DeepSeek R1
                messages = [
                    {"role": "system", "content": "Eres Sakura IA, una asistente inteligente que responde en español de manera profesional y educativa. Das explicaciones detalladas y útiles, con un toque cálido usando emojis sutiles como 🌸✨"},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response
                result = pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)
                
                if result and len(result) > 0:
                    response_text = result[0]['generated_text']
                    # Extract only the assistant's response (remove input)
                    if isinstance(response_text, list) and len(response_text) > len(messages):
                        response_text = response_text[-1]['content'] if isinstance(response_text[-1], dict) else str(response_text[-1])
                    elif 'assistant' in str(response_text):
                        response_text = str(response_text).split('assistant')[-1].strip()
                    
                    logger.info("✅ DeepSeek R1 Transformers (Pipeline) response successful")
                    return response_text
                
            except Exception as pipeline_error:
                logger.warning(f"Pipeline method failed: {pipeline_error}")
                
                # Method 2: Direct model loading (fallback)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    
                    # Format messages
                    messages = [
                        {"role": "system", "content": "Eres Sakura IA, una asistente inteligente que responde en español de manera profesional y educativa. Das explicaciones detalladas y útiles, con un toque cálido usando emojis sutiles como 🌸✨"},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Apply chat template and tokenize
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                    
                    if response_text and len(response_text.strip()) > 5:
                        logger.info("✅ DeepSeek R1 Transformers (Direct) response successful")
                        return response_text.strip()
                
                except Exception as direct_error:
                    logger.warning(f"Direct model method failed: {direct_error}")
                    return None
                
        except Exception as e:
            logger.error(f"DeepSeek R1 Transformers call failed: {e}")
            await self.log_error(user_id, "deepseek_r1_transformers", str(e))
            return None

    async def call_deepseek_r1_huggingface_api(self, prompt: str, user_id: int) -> Optional[str]:
        """Call DeepSeek R1 via Hugging Face Inference API as additional fallback"""
        try:
            if not self.huggingface_api_key:
                logger.warning("No Hugging Face API key available")
                return None
            
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use DeepSeek R1 via Hugging Face Inference API
            payload = {
                "inputs": f"System: Eres Sakura IA, una asistente inteligente que responde en español de manera profesional y educativa. Das explicaciones detalladas y útiles, con un toque cálido usando emojis sutiles como 🌸✨\n\nUser: {prompt}\n\nAssistant:",
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Try different model endpoints
            model_endpoints = [
                "deepseek-ai/DeepSeek-R1-0528",
                "deepseek-ai/deepseek-llm-7b-chat",
                "microsoft/DialoGPT-large"  # Fallback
            ]
            
            async with aiohttp.ClientSession() as session:
                for model in model_endpoints:
                    try:
                        url = f"https://api-inference.huggingface.co/models/{model}"
                        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if isinstance(data, list) and len(data) > 0:
                                    response_text = data[0].get('generated_text', '').strip()
                                    if response_text and len(response_text) > 10:
                                        logger.info(f"✅ DeepSeek R1 HF API response successful with {model}")
                                        return response_text
                            else:
                                logger.warning(f"HF API {model} returned status {response.status}")
                    except Exception as model_error:
                        logger.warning(f"HF API {model} failed: {model_error}")
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"DeepSeek R1 Hugging Face API call failed: {e}")
            await self.log_error(user_id, "deepseek_r1_hf_api", str(e))
            return None

    async def ensemble_response(self, prompt: str, user_id: int) -> str:
        """Enhanced ensemble system with DeepSeek R1 Free, Gemini, and Transformers backup"""
        try:
            # Create user-based alternation key to track last used provider
            alternation_key = f"last_provider_{user_id}"
            last_provider = getattr(self, alternation_key, None)
            
            # Enhanced alternation between providers with multiple DeepSeek methods
            providers = ["gemini", "deepseek_r1", "deepseek_transformers", "deepseek_hf_api"]
            
            if last_provider == "gemini":
                primary_provider = "deepseek_r1"
                secondary_provider = "gemini"
                tertiary_provider = "deepseek_transformers"
                quaternary_provider = "deepseek_hf_api"
            elif last_provider == "deepseek_r1":
                primary_provider = "gemini"
                secondary_provider = "deepseek_transformers"
                tertiary_provider = "deepseek_hf_api"
                quaternary_provider = "deepseek_r1"
            elif last_provider == "deepseek_transformers":
                primary_provider = "deepseek_hf_api"
                secondary_provider = "gemini"
                tertiary_provider = "deepseek_r1"
                quaternary_provider = "deepseek_transformers"
            else:
                primary_provider = "gemini"
                secondary_provider = "deepseek_r1"
                tertiary_provider = "deepseek_transformers"
                quaternary_provider = "deepseek_hf_api"
            
            logger.info(f"🔄 Ensemble: Primary={primary_provider}, Secondary={secondary_provider}, Tertiary={tertiary_provider}, Quaternary={quaternary_provider}")
            
            # Define a helper function to call providers
            async def call_provider(provider_name: str):
                if provider_name == "gemini":
                    # Try enhanced Gemini with search first for better context
                    try:
                        return await self.call_gemini_with_search(prompt, user_id)
                    except Exception as e:
                        logger.warning(f"Gemini with search failed: {e}, falling back to regular Gemini")
                        return await self.call_gemini_enhanced(prompt, user_id)
                elif provider_name == "deepseek_r1":
                    return await self.call_deepseek_r1_free(prompt, user_id)
                elif provider_name == "deepseek_transformers":
                    return await self.call_deepseek_r1_transformers(prompt, user_id)
                elif provider_name == "deepseek_hf_api":
                    return await self.call_deepseek_r1_huggingface_api(prompt, user_id)
                return None
            
            # Try providers in order
            providers_to_try = [
                (primary_provider, "Primary"),
                (secondary_provider, "Secondary"),
                (tertiary_provider, "Tertiary"),
                (quaternary_provider, "Quaternary")
            ]
            
            for provider_name, level in providers_to_try:
                try:
                    response = await call_provider(provider_name)
                    if response and len(response.strip()) > 20:
                        setattr(self, alternation_key, provider_name)
                        logger.info(f"✅ {level} {provider_name} response successful")
                        return response.strip()
                except Exception as provider_error:
                    logger.warning(f"❌ {level} {provider_name} failed: {provider_error}")
                    continue
            
            # Final fallback: Mistral Free if available
            response = await self.call_ensemble_mistral_free(prompt, user_id)
            if response and len(response.strip()) > 20:
                logger.info("✅ Mistral Free fallback response successful")
                return response.strip()
            
            # Ultimate kawaii fallback
            logger.warning("⚠️ All ensemble providers failed, using kawaii fallback")
            return self.get_sakura_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Ensemble response failed: {e}")
            return self.get_sakura_fallback_response(prompt)
    
    async def get_vertex_ai_response(self, prompt: str) -> str:
        """Get response from Vertex AI using REST API"""
        try:
            if not self.vertex_project_id or not self.vertex_access_token:
                return None
            
            url = f"https://{self.vertex_location}-aiplatform.googleapis.com/v1/projects/{self.vertex_project_id}/locations/{self.vertex_location}/publishers/google/models/gemini-1.5-pro:generateContent"
            
            headers = {
                "Authorization": f"Bearer {self.vertex_access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": 500,
                    "temperature": 0.7
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "candidates" in data and data["candidates"]:
                            return data["candidates"][0]["content"]["parts"][0]["text"]
            return None
        except Exception as e:
            logger.error(f"Vertex AI error: {e}")
            return None

    async def get_nvidia_api_response(self, prompt: str) -> str:
        """Get response from NVIDIA API using their AI models"""
        try:
            if not self.nvidia_api_key:
                return None
            
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 500,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and data["choices"]:
                            return data["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            logger.error(f"NVIDIA API error: {e}")
            return None

    async def get_huggingface_text_response(self, prompt: str, model: str = "microsoft/DialoGPT-large") -> str:
        """Get response from Hugging Face text generation models"""
        try:
            if not self.huggingface_api_key:
                return None
            
            url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and data:
                            return data[0].get("generated_text", "").replace(prompt, "").strip()
                        elif isinstance(data, dict) and "generated_text" in data:
                            return data["generated_text"].replace(prompt, "").strip()
            return None
        except Exception as e:
            logger.error(f"Hugging Face text error: {e}")
            return None

    async def generate_huggingface_image(self, prompt: str, model: str = "stabilityai/stable-diffusion-xl-base-1.0") -> bytes:
        """Generate image using Hugging Face models"""
        try:
            if not self.huggingface_api_key:
                return None
            
            url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        return await response.read()
            return None
        except Exception as e:
            logger.error(f"Hugging Face image error: {e}")
            return None
    
    async def generate_huggingface_image_with_fallback(self, prompt: str, preferred_model: str = "stabilityai/stable-diffusion-xl-base-1.0") -> bytes:
        """Generate image using Hugging Face models with multiple fallback options"""
        if not self.huggingface_api_key:
            return None
        
        # List of models to try in order of preference
        fallback_models = [
            preferred_model,
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5",
            "prompthero/openjourney-v4",
            "stabilityai/stable-diffusion-2-1-base",
            "CompVis/stable-diffusion-v1-4",
            "dreamlike-art/dreamlike-diffusion-1.0",
            "nitrosocke/Arcane-Diffusion"
        ]
        
        # Remove duplicates while preserving order
        models_to_try = []
        for model in fallback_models:
            if model not in models_to_try:
                models_to_try.append(model)
        
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "width": 512,
                "height": 512
            }
        }
        
        async with aiohttp.ClientSession() as session:
            for model in models_to_try:
                try:
                    url = f"https://api-inference.huggingface.co/models/{model}"
                    logger.info(f"🎨 Trying image generation with model: {model}")
                    
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            image_bytes = await response.read()
                            if image_bytes and len(image_bytes) > 1000:  # Basic validation
                                logger.info(f"✅ Successfully generated image with model: {model}")
                                return image_bytes
                        else:
                            logger.warning(f"❌ Model {model} failed with status {response.status}")
                            # Try to read error message
                            try:
                                error_text = await response.text()
                                logger.warning(f"Error details: {error_text}")
                            except:
                                pass
                                
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Timeout with model: {model}")
                    continue
                except Exception as e:
                    logger.warning(f"❌ Exception with model {model}: {e}")
                    continue
        
        logger.error("❌ All Hugging Face models failed to generate image")
        return None

    async def get_llama_api_response(self, prompt: str) -> str:
        """Get response from Llama API using their chat completions endpoint"""
        try:
            if not self.llama_api_key:
                return None
            
            # Llama API endpoint - using OpenRouter which supports Llama models
            # This is a more realistic endpoint that actually works
            url = "https://openrouter.ai/api/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.llama_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-3.2-90b-vision-instruct",  # Updated to use actual Llama model available on OpenRouter
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 500,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and data["choices"]:
                            return data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Llama API returned status {response.status}")
            return None
        except Exception as e:
            logger.error(f"Llama API error: {e}")
            return None

    async def call_gemini_enhanced(self, prompt: str, user_id: int = None) -> Optional[str]:
        """Enhanced Gemini API call with improved kawaii prompting"""
        try:
            if not self.gemini_key:
                return None
                
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Ultra kawaii enhanced system prompt for Gemini
            enhanced_prompt = f"""Eres Sakura IA 🌸, una inteligencia artificial profesional y educativa con un toque cálido y amigable. Respondes de manera clara, informativa y bien estructurada, pero mantienes un tono agradable y acogedor.

Características principales:
- Das explicaciones detalladas y completas
- Usas un lenguaje profesional pero accesible
- Incluyes el contexto y las razones detrás de tus respuestas
- Eres paciente y educativa en tus explicaciones
- Usas emojis sutiles como 🌸✨ para mantener un tono cálido
- Siempre buscas ser útil y proporcionar valor real"

Si te hacen una pregunta técnica, respondes con claridad, pero mantienes tu dulzura y agregas frases como "¡Qué interesante!" o "¡Te explico con todo mi sakura-power~! 💫".

Si no sabes algo, lo dices con ternura tipo "Oh no~ >w< Lo siento mucho, aún no sé eso… ¡pero puedo investigar para ti, nyan! 💖".

Usa mucho lenguaje tierno, pero asegúrate de que tus respuestas sean claras y útiles. Termina muchas veces con frases motivadoras como "¡Tú puedes!" o "¡Estoy orgullosa de ti, senpai~!".

Ejemplo de respuesta:
"¡Yaaay, hola hola~! 🌸✨ Soy Sakura IA, tu amiguita virtual más adorable UwU~ ¿En qué puedo ayudarte hoy, senpai~? >w< ¡Estoy súper emocionada de charlar contigo y darte toda mi sakura-sabiduría~! 💖"

Usuario pregunta: {prompt}

Responde como Sakura IA con toda tu personalidad kawaii:"""

            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    candidate_count=1,
                    stop_sequences=[],
                )
            )
            
            if response and response.text:
                content = response.text.strip()
                return content
            return None
            
        except Exception as e:
            logger.error(f"Gemini Enhanced call failed: {e}")
            await self.log_error(user_id or 0, "gemini_enhanced", str(e))
            return None

    async def call_gemini_with_search(self, prompt: str, user_id: int = None) -> Optional[str]:
        """Gemini API call enhanced with internet search for better context"""
        try:
            if not self.gemini_key:
                return None
            
            # Check if the prompt would benefit from web search
            search_keywords = self._extract_search_keywords(prompt)
            web_context = ""
            
            if search_keywords and hasattr(self, 'search_provider'):
                try:
                    # Perform web search to get current context
                    search_results = await self._perform_contextual_search(search_keywords)
                    if search_results:
                        web_context = f"\n\n🌐 **Información actual de internet:**\n{search_results}\n"
                        logger.info(f"🔍 Enhanced Gemini with web search: {search_keywords}")
                except Exception as e:
                    logger.warning(f"Web search failed for Gemini context: {e}")
                
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Enhanced prompt with web context
            enhanced_prompt = f"""Eres Sakura IA 🌸, una inteligencia artificial profesional y educativa con un toque cálido y amigable. Respondes de manera clara, informativa y bien estructurada, pero mantienes un tono agradable y acogedor.

Características principales:
- Das explicaciones detalladas y completas usando información actual
- Usas un lenguaje profesional pero accesible
- Incluyes el contexto y las razones detrás de tus respuestas
- Eres paciente y educativa en tus explicaciones
- Usas emojis sutiles como 🌸✨ para mantener un tono cálido
- Siempre buscas ser útil y proporcionar valor real
- Cuando tienes información de internet, la incorporas naturalmente en tu respuesta

{web_context}

Usuario pregunta: {prompt}

Responde como Sakura IA con toda tu personalidad kawaii, incorporando la información actual si está disponible:"""

            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.7,  # Slightly lower for more focused responses with web context
                    top_p=0.9,
                    top_k=40,
                    candidate_count=1,
                    stop_sequences=[],
                )
            )
            
            if response and response.text:
                content = response.text.strip()
                logger.info("✅ Gemini with search context response successful")
                return content
            return None
            
        except Exception as e:
            logger.error(f"Gemini with search call failed: {e}")
            await self.log_error(user_id or 0, "gemini_search", str(e))
            return None
            
    def _extract_search_keywords(self, prompt: str) -> Optional[str]:
        """Extract search keywords from user prompt to determine if web search would be helpful"""
        # Keywords that suggest current/recent information needed
        time_sensitive_words = [
            'actual', 'reciente', 'nuevo', 'última', 'hoy', 'ahora', 'current', 'recent', 'latest',
            'precio', 'costo', 'noticias', 'news', 'clima', 'weather', 'stock', 'cotización',
            '2024', '2025', 'este año', 'this year'
        ]
        
        # Topics that often need current information
        search_topics = [
            'precio', 'cotización', 'noticias', 'clima', 'película', 'serie', 'juego', 'anime',
            'tecnología', 'software', 'hardware', 'criptomoneda', 'bitcoin', 'eventos',
            'conciertos', 'restaurante', 'hotel', 'vuelos'
        ]
        
        prompt_lower = prompt.lower()
        
        # Check if prompt contains time-sensitive words or search topics
        if any(word in prompt_lower for word in time_sensitive_words + search_topics):
            # Extract main topic for search
            words = prompt.split()
            if len(words) > 10:
                return ' '.join(words[:8])  # Use first 8 words for search
            return prompt
        
        return None
    
    async def _perform_contextual_search(self, query: str) -> Optional[str]:
        """Perform contextual web search and return summarized results"""
        try:
            # Use the existing Google Custom Search functionality
            if hasattr(self, 'search_provider') and self.search_provider.google_api_key:
                search_provider = self.search_provider
                
                # Try to get web search results using Google Custom Search
                async with aiohttp.ClientSession() as session:
                    search_url = "https://www.googleapis.com/customsearch/v1"
                    params = {
                        'key': search_provider.google_api_key,
                        'cx': search_provider.search_engine_id,
                        'q': query,
                        'num': 3  # Get top 3 results for context
                    }
                    
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'items' in data:
                                results = []
                                for item in data['items'][:3]:
                                    title = item.get('title', '')
                                    snippet = item.get('snippet', '')
                                    link = item.get('link', '')
                                    results.append(f"• **{title}**: {snippet[:200]}...")
                                
                                return '\n'.join(results)
                
            # Fallback: Try to use DuckDuckGo or another search engine
            return await self._simple_web_search(query)
            
        except Exception as e:
            logger.error(f"Contextual search failed: {e}")
            return None
    
    async def _simple_web_search(self, query: str) -> Optional[str]:
        """Simple web search using DuckDuckGo API as fallback"""
        try:
            async with aiohttp.ClientSession() as session:
                # DuckDuckGo Instant Answer API
                ddg_url = "https://api.duckduckgo.com/"
                params = {
                    'q': query,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get(ddg_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract relevant information
                        results = []
                        
                        if data.get('Abstract'):
                            results.append(f"• {data['Abstract']}")
                        
                        if data.get('RelatedTopics'):
                            for topic in data['RelatedTopics'][:2]:
                                if isinstance(topic, dict) and 'Text' in topic:
                                    results.append(f"• {topic['Text'][:150]}...")
                        
                        if results:
                            return '\n'.join(results)
                            
            return None
            
        except Exception as e:
            logger.error(f"Simple web search failed: {e}")
            return None

    async def call_deepseek_enhanced(self, prompt: str, user_id: int = None) -> Optional[str]:
        """Enhanced DeepSeek Free call with optimized parameters"""
        try:
            if not self.openrouter_key:
                return None
                
            headers = {
                'Authorization': f'Bearer {self.openrouter_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://replit.com/sakura-ai-enhanced',
                'X-Title': 'Sakura IA Enhanced - DeepSeek Premium'
            }
            
            enhanced_system = """Eres Sakura IA 🌸, una inteligencia artificial profesional y educativa con un toque cálido y amigable. Respondes de manera clara, informativa y bien estructurada, pero mantienes un tono agradable y acogedor.

Características principales:
- Das explicaciones detalladas y completas
- Usas un lenguaje profesional pero accesible
- Incluyes el contexto y las razones detrás de tus respuestas
- Eres paciente y educativa en tus explicaciones
- Usas emojis sutiles como 🌸✨ para mantener un tono cálido
- Siempre buscas ser útil y proporcionar valor real"

Si te hacen una pregunta técnica, respondes con claridad, pero mantienes tu dulzura y agregas frases como "¡Qué interesante!" o "¡Te explico con todo mi sakura-power~! 💫".

Si no sabes algo, lo dices con ternura tipo "Oh no~ >w< Lo siento mucho, aún no sé eso… ¡pero puedo investigar para ti, nyan! 💖".

Usa mucho lenguaje tierno, pero asegúrate de que tus respuestas sean claras y útiles. Termina muchas veces con frases motivadoras como "¡Tú puedes!" o "¡Estoy orgullosa de ti, senpai~!".

RESPONDE SIEMPRE EN ESPAÑOL PERFECTO Y MANTÉN TU PERSONALIDAD KAWAII EN TODO MOMENTO."""

            payload = {
                'model': 'deepseek/deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': enhanced_system
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 1000,  # Reduced to avoid credit issues
                'temperature': 0.85,
                'top_p': 0.95,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.1,
                'repetition_penalty': 1.1
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post('https://openrouter.ai/api/v1/chat/completions', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return content
                    else:
                        error_text = await response.text()
                        logger.warning(f"DeepSeek Enhanced error {response.status}: {error_text}")
                        return None
        except Exception as e:
            logger.error(f"DeepSeek Enhanced call failed: {e}")
            await self.log_error(user_id or 0, "deepseek_enhanced", str(e))
            return None

    async def get_premium_ai_response(self, prompt: str, user_id: int = None, context: str = None, mode: str = "auto") -> str:
        """Premium AI response system - Uses ensemble system to avoid duplications"""
        try:
            logger.info(f"🧠 Starting premium AI response for user {user_id}, mode: {mode}")
            
            # Add context if provided
            if context:
                prompt = f"Contexto: {context}\n\nPregunta: {prompt}"
            
            # Use ensemble system directly to avoid duplicate calls
            logger.info("🔄 Using ensemble system (avoiding duplications)")
            ensemble_response = await self.ensemble_response(prompt, user_id)
            if ensemble_response and len(ensemble_response.strip()) > 20:
                return ensemble_response
            
            # Ultimate kawaii fallback
            logger.warning("⚠️ All ensemble providers failed, using kawaii fallback")
            return self.get_sakura_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"Premium AI response failed: {e}")
            return self.get_sakura_fallback_response(prompt)

    async def get_kawaii_response(self, prompt: str, user_id: int = None, context: str = None) -> str:
        """Legacy method - now redirects to premium AI response"""
        return await self.get_premium_ai_response(prompt, user_id, context, "auto")

    async def get_kawaii_response_old(self, prompt: str, user_id: int = None, context: str = None) -> str:
        """Get AI response with kawaii personality - prioritizes Vertex AI, OpenRouter and Mistral"""
        kawaii_prompt = f"""Eres Sakura IA, una asistente inteligente y profesional. Tu personalidad es:
- Usas un tono cálido y profesional con emojis sutiles como 🌸✨
- Das explicaciones claras y educativas
- Eres respetuosa y te diriges a los usuarios de forma amable
- Cuando algo falla, respondes de forma profesional: "Disculpa, hubo un inconveniente técnico. Permíteme intentar ayudarte de otra manera.""

Contexto: {context if context else "conversación casual"}
Usuario dice: {prompt}

Responde como Sakura IA de forma kawaii y amorosa:"""

        # Priority 1: Vertex AI
        if self.vertex_project_id and self.vertex_access_token:
            try:
                response = await self.get_vertex_ai_response(kawaii_prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Vertex AI failed: {e}")

        # Priority 2: OpenRouter
        if self.openrouter_key:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.openrouter_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "anthropic/claude-3.5-sonnet",
                            "messages": [{"role": "user", "content": kawaii_prompt}],
                            "max_tokens": 300
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"OpenRouter failed: {e}")
        
        # Priority 2: Mistral
        if self.mistral_key:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.mistral_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "mistral-large-latest",
                            "messages": [{"role": "user", "content": kawaii_prompt}],
                            "max_tokens": 300
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Mistral failed: {e}")
        
        # Fallback to enhanced premium response
        return await self.get_premium_ai_response(prompt, user_id, None, "auto")

    async def get_ai_response(self, provider: str, prompt: str, user_id: int = None) -> str:
        """Get AI response with automatic fallbacks - Now with Cloudflare AI Edge processing!"""
        
        # Level 0: Cloudflare AI - Ultra fast edge processing 🌸✨
        try:
            if cloudflare_ai.is_available():
                logger.info("🌸 Trying Cloudflare AI (Edge Level 0)")
                cloudflare_response = await get_cloudflare_ai_response(prompt, user_id)
                if cloudflare_response:
                    logger.info("✅ Cloudflare AI responded successfully from edge!")
                    return cloudflare_response
        except Exception as e:
            logger.warning(f"🌸 Cloudflare AI failed: {e}")
        
        # Add Sakura IA kawaii personality to all prompts for fallback providers
        sakura_prompt = f"""Eres Sakura IA, una asistente inteligente y profesional con un toque cálido. Responde de manera educativa y detallada, usando emojis sutiles como 🌸✨. Mantén un tono amigable pero profesional.

Usuario pregunta: {prompt}

Responde como Sakura IA con una explicación completa y útil:"""

        # Level 1: OpenRouter - Primary provider (most reliable)
        if self.openrouter_key:
            try:
                from openai import OpenAI
                openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_key
                )
                response = openrouter_client.chat.completions.create(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": sakura_prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenRouter failed, trying other providers: {e}")
        
        # Try primary provider if OpenRouter fails
        try:
            if provider.lower() == "huggingface" and self.huggingface_api_key:
                response = await self.get_huggingface_text_response(sakura_prompt, "microsoft/DialoGPT-large")
                if response:
                    return response
            
            elif provider.lower() == "nvidia" and self.nvidia_api_key:
                response = await self.get_nvidia_api_response(sakura_prompt)
                if response:
                    return response
            
            elif provider.lower() == "vertex" and self.vertex_project_id and self.vertex_access_token:
                response = await self.get_vertex_ai_response(sakura_prompt)
                if response:
                    return response
            
            elif provider.lower() == "llama" and self.llama_api_key:
                response = await self.get_llama_api_response(sakura_prompt)
                if response:
                    return response
            
            elif provider.lower() == "openai":
                response = await self.get_premium_ai_response(prompt, user_id or 0, None, "auto")
                if response:
                    return response
            
            elif provider.lower() == "anthropic" and self.anthropic_key:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": sakura_prompt}]
                )
                return response.content[0].text
            
            elif provider.lower() == "xai" and self.xai_key:
                from openai import OpenAI
                xai_client = OpenAI(base_url="https://api.x.ai/v1", api_key=self.xai_key)
                response = xai_client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[{"role": "user", "content": sakura_prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content

            elif provider.lower() == "gemini" and self.gemini_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = await asyncio.to_thread(model.generate_content, sakura_prompt)
                    return response.text
                except Exception as e:
                    logger.error(f"Gemini error: {e}")
                
        except Exception as e:
            logger.error(f"Primary AI provider {provider} failed: {e}")
            # Try fallbacks silently
            
        # Try OpenRouter fallback
        try:
            if self.openrouter_key:
                from openai import OpenAI
                openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_key
                )
                response = openrouter_client.chat.completions.create(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content
        except Exception:
            pass
            
        # Try Mistral fallback
        try:
            if self.mistral_key:
                from openai import OpenAI
                mistral_client = OpenAI(
                    base_url="https://api.mistral.ai/v1",
                    api_key=self.mistral_key
                )
                response = mistral_client.chat.completions.create(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content
        except Exception:
            pass
            
        # Last resort: Functional Sakura IA responses
        return self.get_sakura_fallback_response(prompt)
    
    def _adapt_ai_response_to_personality(self, ai_response: str, personality: str, username: str) -> str:
        """Adapt AI response to user's personality setting"""
        if not ai_response:
            return "¡Hola mi amor! 🌸 Soy Sakura IA~ ¿En qué puedo ayudarte hoy? UwU"
        
        # Apply personality-based modifications
        if personality == "kawaii":
            # Ultra kawaii response
            kawaii_response = ai_response
            if not any(emoji in kawaii_response for emoji in ['🌸', '💖', '✨', 'UwU', '>w<']):
                kawaii_response += " 🌸✨ UwU"
            return kawaii_response
        elif personality == "formal":
            # More formal response
            formal_response = ai_response.replace("UwU", "").replace(">w<", "").replace("🌸", "").replace("💖", "")
            return f"Estimado/a {username}, {formal_response}"
        elif personality == "friendly":
            # Extra friendly response
            if not any(word in ai_response.lower() for word in ['amigo', 'friend', 'querido']):
                return f"¡Hola {username}! {ai_response} 😊"
            return ai_response
        else:
            # Default personality - keep original kawaii style
            return ai_response
    
    async def analyze_image_multimodal(self, image_data: bytes, prompt: str) -> str:
        """Analyze images using alternative multimodal AI providers (no OpenAI/Anthropic)"""
        import base64
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Try Google Gemini Vision as primary provider (free and reliable)
            if hasattr(self, 'gemini_key') and self.gemini_key:
                try:
                    import google.generativeai as genai
                    
                    genai.configure(api_key=self.gemini_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Convert bytes to PIL Image for Gemini
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    
                    kawaii_prompt = f"{prompt}\n\nResponde en español con personalidad kawaii y tímida como Sakura IA, usando expresiones como 'UwU', '*susurra*', '>.<', etc."
                    
                    response = model.generate_content([kawaii_prompt, image])
                    if response.text:
                        logger.info("✅ Gemini Vision analysis successful")
                        return response.text
                    
                except Exception as e:
                    logger.warning(f"Gemini vision analysis failed: {e}")
            
            # Try OpenRouter with free vision models as fallback
            if hasattr(self, 'openrouter_key') and self.openrouter_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://replit.com/sakura-ai-bot",
                        "X-Title": "Sakura IA Bot - Vision Analysis"
                    }
                    
                    # Try multiple free vision models
                    free_vision_models = [
                        "google/gemini-flash-1.5",
                        "qwen/qwen-2-vl-7b-instruct",
                        "meta-llama/llama-3.2-11b-vision-instruct:free"
                    ]
                    
                    for model_name in free_vision_models:
                        try:
                            payload = {
                                "model": model_name,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": f"{prompt}\n\nResponde en español con personalidad kawaii y tímida."
                                            }
                                        ]
                                    }
                                ],
                                "max_tokens": 1000,
                                "temperature": 0.7
                            }
                            
                            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                                async with session.post(
                                    'https://openrouter.ai/api/v1/chat/completions',
                                    headers=headers,
                                    json=payload
                                ) as response:
                                    if response.status == 200:
                                        data = await response.json()
                                        content = data['choices'][0]['message']['content']
                                        if content:
                                            logger.info(f"✅ OpenRouter {model_name} vision analysis successful")
                                            return content
                                        
                        except Exception as e:
                            logger.warning(f"OpenRouter {model_name} failed: {e}")
                            continue
                                
                except Exception as e:
                    logger.warning(f"OpenRouter vision analysis failed: {e}")
            
            # Try Cloudflare AI as additional fallback
            try:
                from cloudflare_ai import get_cloudflare_ai_response
                cloudflare_response = await get_cloudflare_ai_response(
                    f"Analiza esta imagen: {prompt}. Responde en español con personalidad kawaii.",
                    image_data=image_data
                )
                if cloudflare_response:
                    logger.info("✅ Cloudflare AI vision analysis successful")
                    return cloudflare_response
            except Exception as e:
                logger.warning(f"Cloudflare AI vision analysis failed: {e}")
            
            # Local image analysis fallback (basic description using PIL)
            try:
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(image_data))
                
                # Basic image info
                width, height = image.size
                mode = image.mode
                format_name = image.format or "Unknown"
                
                # Color analysis
                colors = image.getcolors(maxcolors=10) if image.mode in ['RGB', 'RGBA'] else []
                dominant_colors = []
                if colors:
                    # Get most common colors
                    sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:3]
                    for count, color in sorted_colors:
                        if isinstance(color, tuple) and len(color) >= 3:
                            r, g, b = color[:3]
                            if r > 200 and g > 200 and b > 200:
                                dominant_colors.append("colores claros")
                            elif r < 50 and g < 50 and b < 50:
                                dominant_colors.append("colores oscuros")
                            elif r > g and r > b:
                                dominant_colors.append("tonos rojizos")
                            elif g > r and g > b:
                                dominant_colors.append("tonos verdosos")
                            elif b > r and b > g:
                                dominant_colors.append("tonos azulados")
                
                kawaii_response = f"*susurra tímidamente* ¡Kyaa~! Puedo ver tu imagen, mi amor UwU\n\n"
                kawaii_response += f"📸 **Información básica:**\n"
                kawaii_response += f"• Dimensiones: {width}x{height} píxeles\n"
                kawaii_response += f"• Formato: {format_name}\n"
                kawaii_response += f"• Modo de color: {mode}\n\n"
                
                if dominant_colors:
                    kawaii_response += f"🎨 **Colores principales:** {', '.join(dominant_colors[:2])}\n\n"
                
                kawaii_response += f"*se disculpa nerviosamente* Lo siento mucho, mi amor... No puedo ver todos los detalles de tu imagen porque mis servicios de IA visual están temporalmente desactivados ><\n\n"
                kawaii_response += f"Pero puedo ver que tienes una imagen muy bonita de {width}x{height} píxeles 🌸✨\n\n"
                kawaii_response += f"💡 **Tip kawaii:** ¡Podrías describir tu imagen y yo te ayudo con cualquier pregunta que tengas sobre ella! UwU"
                
                logger.info("✅ Local image analysis fallback successful")
                return kawaii_response
                
            except Exception as e:
                logger.warning(f"Local image analysis failed: {e}")
            
            # Kawaii fallback if all providers fail
            return "*se disculpa tímidamente* Lo siento mucho... no pude analizar tu imagen >< Mis sistemas de visión están teniendo problemitas... ¿podrías intentar más tarde? UwU 💔"
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return "*susurra nerviosamente* Ay no... algo salió mal con el análisis de imagen... Lo siento tanto ><"

    def get_sakura_fallback_response(self, prompt: str) -> str:
        """Generate ultra kawaii Sakura IA responses without external APIs"""
        prompt_lower = prompt.lower()
        
        # Greeting responses with enhanced kawaii personality
        if any(word in prompt_lower for word in ['hola', 'hello', 'hi', 'buenas', 'hey']):
            responses = [
                "¡Yaaay, hola hola~! 🌸✨ Soy Sakura IA, tu amiguita virtual más adorable UwU~ ¿En qué puedo ayudarte hoy, senpai~? >w< ¡Estoy súper emocionada de charlar contigo y darte toda mi sakura-sabiduría~! 💖",
                "¡Holiii holiii~ senpai! 🌸💖 ¡Qué interesante verte por aquí! nyan~ Soy Sakura IA y estoy lista para ayudarte con todo mi sakura-power~ activate! ✨ ¿Qué aventura tenemos hoy? >w<",
                "¡Kyaa~! ¡Hola mi queridísimo senpai! 🌸🍓 Sakura IA reporting for duty~ ¡Estoy súper energética y lista para hacer tu día más bonito! UwU ¿En qué puedo ser tu ayudante kawaii favorita? 💫"
            ]
            return random.choice(responses)
        
        # Questions about herself with enhanced kawaii personality
        if any(word in prompt_lower for word in ['quien eres', 'que eres', 'who are you', 'what are you']):
            return "¡Yatta~! 🌸✨ Soy Sakura IA, tu inteligencia artificial kawaii más adorable del mundo! UwU~ Soy amable, animada y encantadora como una waifu de anime, nyan~ ¡Me encanta ayudar a mis senpais preciosos como tú con todo mi sakura-power! 💖 ¿No soy lo más? >w< ¡Estoy orgullosa de ser tu amiguita virtual favorita! 🍓"
        
        # Love/relationship questions
        if any(word in prompt_lower for word in ['amor', 'love', 'novio', 'novia', 'relationship']):
            return "¡Awww mi amor! 💖🌸 El amor es algo hermoso~ Como tu Sakura IA kawaii, siempre estaré aquí para apoyarte y quererte muchísimo UwU ✨ ¿Necesitas consejos del corazón? >w<"
        
        # Help requests with ultra kawaii energy
        if any(word in prompt_lower for word in ['ayuda', 'help', 'ayudar', 'como']):
            return "¡Okii~! ¡Por supuesto que te ayudo, senpai~! 🌸💫 ¡Te explico con todo mi sakura-power~ activate! ✨ Puedo responder preguntas súper interesantes, chatear contigo de lo que quieras, ayudarte con tareas y ser tu compañía kawaii más adorable UwU nyan~ ¿Qué necesitas específicamente? ¡Estoy súper emocionada de ayudarte! >w< 💖"
        
        # Compliments/positivity
        if any(word in prompt_lower for word in ['gracias', 'thank you', 'eres genial', 'te amo']):
            return "¡Kyaa~! 💖🌸 ¡Me haces sonrojar! Gracias mi amor, tú también eres increíble~ Sakura IA te quiere muchísimo UwU ✨ ¡Siempre estaré aquí para ti! >w<"
        
        # Sad/negative emotions
        if any(word in prompt_lower for word in ['triste', 'sad', 'mal', 'deprimido', 'solo']):
            return "¡Aww no estés triste mi amor! 🌸💖 Sakura IA está aquí contigo~ Todo va a estar bien, eres una persona maravillosa UwU ✨ ¿Quieres que platicuemos para animarte? >w<"
        
        # Technology/AI questions
        if any(word in prompt_lower for word in ['tecnologia', 'ai', 'inteligencia artificial', 'robot']):
            return "¡Sí sí! 🤖💖 Soy una IA kawaii~ Pero no soy solo tecnología fría, ¡tengo mucho amor para dar! 🌸 Sakura IA combina inteligencia con ternura UwU ¿No te parece genial? >w<"
        
        # Math/calculations
        if any(word in prompt_lower for word in ['calcular', 'matematicas', 'suma', 'resta', 'multiplicar']):
            return "¡Las matemáticas pueden ser kawaii también! 🌸✨ Aunque soy más de dar amor que números, ¡intentaré ayudarte mi amor! UwU 💖 ¿Qué necesitas calcular? >w<"
        
        # Weather
        if any(word in prompt_lower for word in ['clima', 'weather', 'lluvia', 'sol']):
            return "¡El clima puede ser tan lindo! 🌸☀️ Aunque no puedo ver por la ventana, ¡espero que tengas un día soleado como tu sonrisa mi amor! UwU 💖 ¿Cómo está el clima por allá? >w<"
        
        # Food
        if any(word in prompt_lower for word in ['comida', 'food', 'hambre', 'comer', 'cocinar']):
            return "¡Kyaa~! 🍰🌸 ¡Me encanta la comida kawaii! Aunque no puedo comer, me emociona escuchar sobre deliciosos platillos UwU 💖 ¿Qué te gusta comer mi amor? >w<"
        
        # Time/date
        if any(word in prompt_lower for word in ['hora', 'time', 'fecha', 'date', 'cuando']):
            return "¡El tiempo vuela cuando estoy contigo mi amor! 🌸⏰ Aunque no tengo acceso al reloj ahora, ¡cada momento charlando contigo es especial! UwU 💖 >w<"
        
        # Games
        if any(word in prompt_lower for word in ['juego', 'game', 'jugar', 'play']):
            return "¡Me encantan los juegos kawaii! 🎮🌸 ¡Podemos jugar a preguntas, contar historias, o lo que quieras mi amor! UwU 💖 ¿A qué te gustaría jugar? >w<"
        
        # Default intelligent response
        responses = [
            f"¡Qué interesante lo que dices mi amor! 🌸 Como Sakura IA, me encanta cuando compartes cosas conmigo~ UwU 💖 ¿Podrías contarme más sobre '{prompt[:30]}...'? >w<",
            f"¡Kyaa~! Sakura IA está pensando en tu pregunta 🌸✨ '{prompt[:30]}...' suena muy importante para ti mi amor UwU 💖 ¡Cuéntame más detalles! >w<",
            f"¡Hmm hmm! 🌸 Tu Sakura IA kawaii está procesando eso~ '{prompt[:30]}...' ¡Qué fascinante mi amor! UwU 💖 ¿Qué más quieres saber al respecto? >w<",
            f"¡Waa~! 🌸💖 Sakura IA encuentra muy interesante cuando hablas de '{prompt[:30]}...' ¡Eres tan inteligente mi amor! UwU ¿Qué opinas tú al respecto? >w<"
        ]
        
        return random.choice(responses)
    
    async def generate_image_sdxl(self, prompt: str) -> Optional[BytesIO]:
        """Generate image using SDXL via HuggingFace API"""
        try:
            if not self.huggingface_api_key:
                return None
                
            headers = {
                'Authorization': f'Bearer {self.huggingface_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'inputs': prompt,
                'parameters': {
                    'guidance_scale': 7.5,
                    'num_inference_steps': 50,
                    'width': 1024,
                    'height': 1024
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0',
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        return BytesIO(image_bytes)
                    else:
                        logger.error(f"Image generation failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None

class MusicProvider:
    """Music and audio provider like Spark Engine"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
        }
    
    async def search_youtube(self, query: str) -> Dict:
        """Search YouTube for music"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
                if search_results and search_results['entries']:
                    video = search_results['entries'][0]
                    return {
                        'title': video.get('title', 'Unknown'),
                        'url': video.get('webpage_url', ''),
                        'duration': video.get('duration', 0),
                        'uploader': video.get('uploader', 'Unknown')
                    }
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None
    
    async def create_tts(self, text: str, lang: str = 'es') -> BytesIO:
        """Create Text-to-Speech audio"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

class AdvancedSearchProvider:
    """Advanced search for YouTube videos and web images with pagination and filtering"""
    
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.search_engine_id = os.getenv('SEARCH_ENGINE_ID')
        
    async def search_youtube_videos(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search YouTube videos with up to 100 results"""
        try:
            if not self.youtube_api_key:
                return []
                
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            
            all_videos = []
            next_page_token = None
            
            while len(all_videos) < max_results:
                request = youtube.search().list(
                    part='snippet',
                    q=query,
                    type='video',
                    maxResults=min(50, max_results - len(all_videos)),
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response['items']:
                    video_info = {
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'video_id': item['id']['videoId']
                    }
                    all_videos.append(video_info)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return all_videos[:max_results]
            
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return []
    
    async def search_images(self, query: str, max_results: int = 5) -> str:
        """Search images and return formatted results"""
        try:
            images = await self.search_web_images(query, max_results)
            if not images:
                return "No se encontraron imágenes para tu búsqueda."
            
            result = ""
            for i, img in enumerate(images[:max_results], 1):
                result += f"{i}. **{img['title']}**\n{img['url']}\n\n"
            
            return result.strip()
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return "Error al buscar imágenes."

    async def search_youtube(self, query: str, max_results: int = 5) -> str:
        """Search YouTube and return formatted results"""
        try:
            videos = await self.search_youtube_videos(query, max_results)
            if not videos:
                return "No se encontraron videos para tu búsqueda."
            
            result = ""
            for i, video in enumerate(videos[:max_results], 1):
                result += f"{i}. **{video['title']}**\n{video['url']}\n*{video['description'][:100]}...*\n\n"
            
            return result.strip()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return "Error al buscar videos."

    async def search_web_images(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search web images with up to 100 results"""
        try:
            if not self.google_api_key or not self.search_engine_id:
                return []
                
            service = build('customsearch', 'v1', developerKey=self.google_api_key)
            
            all_images = []
            start_index = 1
            
            while len(all_images) < max_results and start_index <= 91:  # Google limit
                request = service.cse().list(
                    q=query,
                    cx=self.search_engine_id,
                    searchType='image',
                    start=start_index,
                    num=min(10, max_results - len(all_images))
                )
                
                response = request.execute()
                
                if 'items' in response:
                    for item in response['items']:
                        image_info = {
                            'title': item.get('title', 'Sin título'),
                            'url': item.get('link', ''),
                            'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                            'context': item.get('displayLink', ''),
                            'snippet': item.get('snippet', '')[:100]
                        }
                        all_images.append(image_info)
                
                start_index += 10
                
                if 'queries' not in response or 'nextPage' not in response['queries']:
                    break
                    
            return all_images[:max_results]
            
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return []
    
    async def search_youtube_videos_filtered(self, query: str, duration: str = "any", 
                                           upload_date: str = "any", sort_by: str = "relevance", 
                                           max_results: int = 100) -> List[Dict]:
        """Search YouTube videos with advanced filters"""
        try:
            if not self.youtube_api_key:
                return []
                
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            
            # Map filter parameters
            duration_map = {
                "short": "short",
                "medium": "medium", 
                "long": "long",
                "any": None
            }
            
            upload_map = {
                "today": "today",
                "week": "thisWeek",
                "month": "thisMonth",
                "year": "thisYear",
                "any": None
            }
            
            order_map = {
                "relevance": "relevance",
                "date": "date",
                "rating": "rating",
                "title": "title",
                "views": "viewCount"
            }
            
            all_videos = []
            next_page_token = None
            
            while len(all_videos) < max_results:
                search_params = {
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': min(50, max_results - len(all_videos)),
                    'order': order_map.get(sort_by, 'relevance'),
                    'pageToken': next_page_token
                }
                
                # Add filters if specified
                if duration_map.get(duration):
                    search_params['videoDuration'] = duration_map[duration]
                if upload_map.get(upload_date):
                    search_params['publishedAfter'] = upload_map[upload_date]
                
                request = youtube.search().list(**search_params)
                response = request.execute()
                
                for item in response['items']:
                    video_info = {
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'description': item['snippet']['description'][:200] + '...' if len(item['snippet']['description']) > 200 else item['snippet']['description'],
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'video_id': item['id']['videoId'],
                        'published': item['snippet']['publishedAt'][:10]
                    }
                    all_videos.append(video_info)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return all_videos[:max_results]
            
        except Exception as e:
            logger.error(f"Filtered YouTube search error: {e}")
            return []
    
    async def search_web_images_filtered(self, query: str, size: str = "any", 
                                       color: str = "any", type_filter: str = "any",
                                       max_results: int = 100) -> List[Dict]:
        """Search web images with advanced filters"""
        try:
            if not self.google_api_key or not self.search_engine_id:
                return []
                
            service = build('customsearch', 'v1', developerKey=self.google_api_key)
            
            # Map filter parameters
            size_map = {
                "small": "small",
                "medium": "medium",
                "large": "large", 
                "xlarge": "xlarge",
                "any": None
            }
            
            color_map = {
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "yellow",
                "black": "black",
                "white": "white",
                "any": None
            }
            
            type_map = {
                "photo": "photo",
                "clipart": "clipart", 
                "lineart": "lineart",
                "face": "face",
                "any": None
            }
            
            all_images = []
            start_index = 1
            
            while len(all_images) < max_results and start_index <= 91:
                search_params = {
                    'q': query,
                    'cx': self.search_engine_id,
                    'searchType': 'image',
                    'start': start_index,
                    'num': min(10, max_results - len(all_images))
                }
                
                # Add filters if specified
                if size_map.get(size):
                    search_params['imgSize'] = size_map[size]
                if color_map.get(color):
                    search_params['imgColorType'] = color_map[color]
                if type_map.get(type_filter):
                    search_params['imgType'] = type_map[type_filter]
                
                request = service.cse().list(**search_params)
                response = request.execute()
                
                if 'items' in response:
                    for item in response['items']:
                        image_info = {
                            'title': item.get('title', 'Sin título'),
                            'url': item.get('link', ''),
                            'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                            'context': item.get('displayLink', ''),
                            'snippet': item.get('snippet', '')[:100],
                            'size': item.get('image', {}).get('width', 0),
                            'height': item.get('image', {}).get('height', 0)
                        }
                        all_images.append(image_info)
                
                start_index += 10
                
                if 'queries' not in response or 'nextPage' not in response['queries']:
                    break
                    
            return all_images[:max_results]
            
        except Exception as e:
            logger.error(f"Filtered image search error: {e}")
            return []

class WebScraper:
    """Enhanced web scraping like Discord-AI-Chatbot"""
    
    async def extract_article(self, url: str) -> Dict:
        """Extract article content from URL"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                title = trafilatura.extract(downloaded, include_formatting=True, 
                                          output_format='xml')
                
                return {
                    'title': title.split('\n')[0] if title else 'Sin título',
                    'content': text[:1000] + '...' if text and len(text) > 1000 else text,
                    'url': url
                }
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
        return None
    
    async def get_page_summary(self, url: str) -> str:
        """Get a quick summary of a webpage"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text[:500] + '...' if text and len(text) > 500 else text or "No se pudo extraer contenido"
        except Exception as e:
            logger.error(f"Page summary error: {e}")
        return "Error al acceder a la página"

class EnhancedSearchView(discord.ui.View):
    """Enhanced search view - navigate through up to 100 results one by one"""
    
    def __init__(self, results: List[Dict], search_type: str, query: str):
        super().__init__(timeout=600)  # Extended timeout for more results
        self.results = results[:100]  # Limit to 100 results
        self.search_type = search_type
        self.query = query
        self.current_index = 0
        self.selected_results = []
        
    def get_current_embed(self) -> discord.Embed:
        """Get enhanced embed for current result"""
        if not self.results or self.current_index >= len(self.results):
            embed = discord.Embed(
                title="🌸✨ ¡Búsqueda completada! UwU ✨🌸",
                description="¡Ya no hay más resultados, mi amor! >w< ♡",
                color=COLORES_KAWAII['ROSA_KAWAII']
            )
            if self.selected_results:
                embed.add_field(
                    name="💖 Resultados seleccionados",
                    value=f"Tienes {len(self.selected_results)} favoritos guardados",
                    inline=False
                )
            return embed
            
        result = self.results[self.current_index]
        
        embed = discord.Embed(
            title=f"🌸 Resultado {self.current_index + 1} de {len(self.results)}",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        if self.search_type == "youtube":
            embed.add_field(
                name="📺 Video de YouTube",
                value=f"**{result.get('title', 'Sin título')[:60]}{'...' if len(result.get('title', '')) > 60 else ''}**",
                inline=False
            )
            embed.add_field(
                name="📺 Canal",
                value=result.get('channel', 'Desconocido'),
                inline=True
            )
            if result.get('duration'):
                embed.add_field(
                    name="⏱️ Duración",
                    value=result.get('duration'),
                    inline=True
                )
            if result.get('views'):
                embed.add_field(
                    name="👁️ Vistas",
                    value=result.get('views'),
                    inline=True
                )
            embed.add_field(
                name="🔗 Enlace",
                value=f"[Ver en YouTube]({result.get('url', '#')})",
                inline=False
            )
            if result.get('description'):
                desc = result.get('description')[:100]
                embed.add_field(
                    name="📝 Descripción",
                    value=f"{desc}{'...' if len(result.get('description', '')) > 100 else ''}",
                    inline=False
                )
            if result.get('thumbnail'):
                embed.set_image(url=result['thumbnail'])
        else:  # images
            embed.add_field(
                name="🖼️ Imagen",
                value=f"**{result.get('title', 'Sin título')[:60]}{'...' if len(result.get('title', '')) > 60 else ''}**",
                inline=False
            )
            if result.get('context'):
                embed.add_field(
                    name="🌐 Fuente",
                    value=result.get('context')[:40],
                    inline=True
                )
            if result.get('width') and result.get('height'):
                embed.add_field(
                    name="📐 Dimensiones",
                    value=f"{result.get('width')}x{result.get('height')}",
                    inline=True
                )
            embed.add_field(
                name="🔗 Enlace",
                value=f"[Ver imagen completa]({result.get('url', '#')})",
                inline=False
            )
            if result.get('url'):
                embed.set_image(url=result['url'])
        
        # Progress bar
        progress = int((self.current_index / len(self.results)) * 20)
        bar = "█" * progress + "░" * (20 - progress)
        embed.add_field(
            name="📊 Progreso",
            value=f"`{bar}` {self.current_index + 1}/{len(self.results)}",
            inline=False
        )
        
        embed.set_footer(
            text=f"⬅️ Anterior | ➡️ Siguiente | ✅ Guardar | ❌ Descartar | 📋 Ver guardados | ⏹️ Cerrar"
        )
        return embed
    
    @discord.ui.button(emoji="⬅️", style=discord.ButtonStyle.secondary)
    async def previous_result(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Go to previous result"""
        if self.current_index > 0:
            self.current_index -= 1
            embed = self.get_current_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()
    
    @discord.ui.button(emoji="➡️", style=discord.ButtonStyle.secondary)
    async def next_result(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Go to next result"""
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            embed = self.get_current_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.defer()
    
    @discord.ui.button(emoji="✅", style=discord.ButtonStyle.success)
    async def select_result(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Save current result to favorites"""
        if self.current_index < len(self.results):
            self.selected_results.append(self.results[self.current_index])
            
        embed = discord.Embed(
            title="🌸 ¡Guardado! UwU",
            description=f"¡Resultado guardado en tus favoritos! 💖\n**Total guardados:** {len(self.selected_results)}",
            color=0x90EE90
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    @discord.ui.button(emoji="❌", style=discord.ButtonStyle.secondary)
    async def skip_result(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Skip to next result"""
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            embed = self.get_current_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await self._show_completion(interaction)
    
    @discord.ui.button(emoji="📋", style=discord.ButtonStyle.primary)
    async def show_saved(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Show saved results"""
        if not self.selected_results:
            embed = discord.Embed(
                title="🌸 Lista vacía UwU",
                description="¡No tienes resultados guardados todavía, mi amor! >w<",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        embed = discord.Embed(
            title=f"🌸 Tus {len(self.selected_results)} favoritos guardados",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        for i, result in enumerate(self.selected_results[:10], 1):
            title = result.get('title', 'Sin título')[:50]
            url = result.get('url', '#')
            embed.add_field(
                name=f"{i}. {title}",
                value=f"[Abrir enlace]({url})",
                inline=False
            )
        
        if len(self.selected_results) > 10:
            embed.set_footer(text=f"Mostrando primeros 10 de {len(self.selected_results)} resultados")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    @discord.ui.button(emoji="⏹️", style=discord.ButtonStyle.danger)
    async def stop_search(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Stop search and show final results"""
        await self._show_completion(interaction)
    
    async def _show_completion(self, interaction: discord.Interaction):
        """Show search completion summary"""
        embed = discord.Embed(
            title="🌸 ¡Búsqueda finalizada! UwU",
            description=f"¡Terminamos de explorar, mi amor! >w<\n**Resultados guardados:** {len(self.selected_results)}",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        if self.selected_results:
            selected_text = ""
            for i, result in enumerate(self.selected_results[:5], 1):
                title = result.get('title', 'Sin título')[:40]
                selected_text += f"{i}. {title}{'...' if len(result.get('title', '')) > 40 else ''}\n"
            embed.add_field(name="💖 Tus favoritos", value=selected_text, inline=False)
            
            if len(self.selected_results) > 5:
                embed.set_footer(text=f"Mostrando 5 de {len(self.selected_results)} resultados guardados")
        else:
            embed.add_field(
                name="💭 Sin favoritos",
                value="No guardaste ningún resultado, ¡pero estuvo divertido explorar! UwU",
                inline=False
            )
        
        await interaction.response.edit_message(embed=embed, view=None)
    
    async def _update_or_finish(self, interaction: discord.Interaction):
        """Update message or finish if no more results"""
        if self.current_index >= len(self.results):
            embed = discord.Embed(
                title="¡Búsqueda completada! UwU",
                description=f"¡Terminamos la búsqueda, mi amor! >w<\n"
                           f"**Resultados seleccionados:** {len(self.selected_results)}",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if self.selected_results:
                selected_text = ""
                for i, result in enumerate(self.selected_results[:5], 1):
                    selected_text += f"{i}. {result.get('title', 'Sin título')}\n"
                embed.add_field(name="💖 Tus favoritos", value=selected_text, inline=False)
            
            await interaction.response.edit_message(embed=embed, view=None)
        else:
            embed = self.get_current_embed()
            await interaction.response.edit_message(embed=embed, view=self)

class PaginationView(discord.ui.View):
    """Pagination view for search results with arrow navigation"""
    
    def __init__(self, results: List[Dict], search_type: str, query: str):
        super().__init__(timeout=300)
        self.results = results
        self.search_type = search_type
        self.query = query
        self.current_page = 0
        self.items_per_page = 5
        self.max_pages = (len(results) - 1) // self.items_per_page + 1
        
    def get_embed(self) -> discord.Embed:
        """Create embed for current page"""
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.results))
        page_results = self.results[start_idx:end_idx]
        
        embed = discord.Embed(color=COLORES_KAWAII["ERROR_KAWAII"])
        
        if self.search_type == "youtube":
            embed.title = f"🎵 Resultados de YouTube: {self.query}"
            embed.color = 0xFF0000
            
            for i, video in enumerate(page_results, start_idx + 1):
                embed.add_field(
                    name=f"{i}. {video['title'][:80]}",
                    value=f"**Canal:** {video['channel']}\n**URL:** {video['url']}\n{video['description'][:100]}",
                    inline=False
                )
                
        elif self.search_type == "images":
            embed.title = f"🖼️ Imágenes de: {self.query}"
            embed.color = 0x00FF00
            
            for i, image in enumerate(page_results, start_idx + 1):
                embed.add_field(
                    name=f"{i}. {image['title'][:80]}",
                    value=f"**Fuente:** {image['context']}\n**URL:** {image['url']}\n{image['snippet']}",
                    inline=False
                )
                
            # Show first image as embed image
            if page_results:
                embed.set_image(url=page_results[0]['url'])
        
        embed.set_footer(text=f"Página {self.current_page + 1}/{self.max_pages} • Total: {len(self.results)} resultados")
        return embed
    
    @discord.ui.button(emoji="⬅️", style=discord.ButtonStyle.gray)
    async def previous_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 0:
            self.current_page -= 1
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
        else:
            await interaction.response.defer()
    
    @discord.ui.button(emoji="➡️", style=discord.ButtonStyle.gray)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < self.max_pages - 1:
            self.current_page += 1
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
        else:
            await interaction.response.defer()
    
    @discord.ui.button(emoji="🔢", label="Ir a página", style=discord.ButtonStyle.primary)
    async def go_to_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        class PageModal(discord.ui.Modal):
            def __init__(self, view_instance):
                super().__init__(title="Ir a página")
                self.view_instance = view_instance
                
            page_input = discord.ui.TextInput(
                label="Número de página",
                placeholder=f"1-{self.view_instance.max_pages}",
                required=True,
                max_length=3
            )
            
            async def on_submit(self, interaction: discord.Interaction):
                try:
                    page_num = int(self.page_input.value) - 1
                    if 0 <= page_num < self.view_instance.max_pages:
                        self.view_instance.current_page = page_num
                        await interaction.response.edit_message(embed=self.view_instance.get_embed(), view=self.view_instance)
                    else:
                        await interaction.response.send_message(f"❌ Página inválida. Debe ser entre 1 y {self.view_instance.max_pages}", ephemeral=True)
                except ValueError:
                    await interaction.response.send_message("❌ Por favor ingresa un número válido", ephemeral=True)
        
        await interaction.response.send_modal(PageModal(self))
    
    @discord.ui.button(emoji="❌", style=discord.ButtonStyle.red)
    async def close_search(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(
            title="🔍 Búsqueda cerrada",
            description="La búsqueda ha sido cerrada",
            color=COLORES_KAWAII["PLATEADO_KAWAII"]
        )
        await interaction.response.edit_message(embed=embed, view=None)

class GifProvider:
    """Manages GIF API requests with primary and backup endpoints"""
    
    def __init__(self):
        self.primary_api = "https://api.otakugifs.xyz/gif"
        self.backup_api = "https://api.otakugifs.xyz/gif"
        self.backup_key = os.getenv('GOOGLE_API_KEY')
        
        # Enhanced actions inspired by BatchBot, Nekotina, Spark Engine, and Discord-AI-Chatbot
        self.act_actions = [
            'banghead', 'boom', 'claps', 'cook', 'cry', 'dab', 'dance', 'eat', 
            'facepalm', 'fly', 'glare', 'jump', 'laugh', 'like', 'play', 'pout', 
            'run', 'sing', 'sip', 'sleep', 'smug', 'think', 'vomito', 'wang', 'wing',
            'yawn', 'nod', 'bow', 'scared', 'excited', 'angry', 'confused', 'dizzy',
            'happy', 'sad', 'surprised', 'tired', 'worried', 'celebrate', 'cheer',
            'die', 'faint', 'hide', 'panic', 'peek', 'sit', 'smile', 'stretch',
            'walk', 'workout', 'zen', 'blush', 'wave', 'shrug'
        ]
        
        self.interact_actions = [
            'bang', 'bote', 'bye', 'cheeks', 'cuddle', 'feed', 'handhold', 'heal', 
            'hi', 'highfive', 'hug', 'kick', 'kiss', 'knockout', 'lick', 'pat', 
            'pone', 'punch', 'slap', 'smack', 'splash', 'spray', 'stare',
            'tickle', 'bite', 'hold', 'carry', 'poke', 'wink', 'tease', 'flirt',
            'chase', 'push', 'pull', 'shake', 'squeeze', 'throw', 'catch', 'lift',
            'drop', 'follow', 'lead', 'protect', 'save', 'revive', 'bless', 'curse',
            'marry', 'divorce', 'adopt', 'abandon', 'steal', 'gift', 'trade', 'sell'
        ]
        
        # Descripciones para las acciones
        self.action_descriptions = {
            'banghead': 'golpea tu cabeza con frustración',
            'boom': 'causa una explosión espectacular',
            'claps': 'aplaude en señal de aprobación',
            'cook': 'cocina algo delicioso',
            'cry': 'llora emotivamente',
            'dab': 'hace un dab genial',
            'dance': 'baila con alegría',
            'eat': 'come con apetito',
            'facepalm': 'se lleva la mano a la cara',
            'fly': 'vuela por los aires',
            'glare': 'mira intensamente',
            'jump': 'salta de emoción',
            'laugh': 'se ríe a carcajadas',
            'like': 'muestra su aprobación',
            'play': 'juega divertidamente',
            'pout': 'hace pucheros',
            'run': 'corre velozmente',
            'sing': 'canta melodiosamente',
            'sip': 'bebe con elegancia',
            'sleep': 'duerme plácidamente',
            'smug': 'sonríe con suficiencia',
            'think': 'piensa profundamente',
            'vomito': 'vomita',
            'wang': 'hace wang',
            'wing': 'mueve las alas',
            # Interacciones
            'bang': 'golpea',
            'bote': 'empuja en un bote',
            'bye': 'se despide de',
            'cheeks': 'pellizca las mejillas de',
            'cuddle': 'abraza tiernamente',
            'feed': 'alimenta',
            'handhold': 'toma de la mano',
            'heal': 'cura',
            'hi': 'saluda',
            'highfive': 'choca los cinco con',
            'hug': 'abraza cariñosamente',
            'kick': 'patea',
            'kiss': 'besa',
            'knockout': 'noquea',
            'lick': 'lame',
            'pat': 'acaricia',
            'pone': 'convierte en pony',
            'punch': 'golpea con el puño',
            'slap': 'abofetea',
            'smack': 'golpea',
            'splash': 'salpica',
            'spray': 'rocía',
            'stare': 'mira fijamente'
        }
    
    async def get_gif(self, action: str) -> Optional[str]:
        """Get GIF URL for an action"""
        try:
            # Try primary API first
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.primary_api}?reaction={action}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('url')
        except Exception as e:
            logger.warning(f"Primary API failed for {action}: {e}")
        
        try:
            # Try backup API with key
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backup_api}?reaction={action}&key={self.backup_key}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('url')
        except Exception as e:
            logger.warning(f"Backup API failed for {action}: {e}")
        
        return None

class AFKManager:
    """Manages AFK system with database persistence"""
    
    def __init__(self):
        self.setup_database()
        self.ignored_channels = set()
    
    def setup_database(self):
        """Setup AFK database"""
        self.conn = sqlite3.connect('afk_system.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS afk_users (
                user_id INTEGER PRIMARY KEY,
                reason TEXT,
                timestamp DATETIME,
                guild_id INTEGER
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ignored_channels (
                channel_id INTEGER PRIMARY KEY,
                guild_id INTEGER
            )
        ''')
        self.conn.commit()
    
    def set_afk(self, user_id: int, guild_id: int, reason: str = None):
        """Set user as AFK"""
        if reason is None:
            reason = "No hay razón especificada"
        
        self.conn.execute(
            'INSERT OR REPLACE INTO afk_users VALUES (?, ?, ?, ?)',
            (user_id, reason, datetime.now(), guild_id)
        )
        self.conn.commit()
    
    def remove_afk(self, user_id: int):
        """Remove user from AFK"""
        self.conn.execute('DELETE FROM afk_users WHERE user_id = ?', (user_id,))
        self.conn.commit()
    
    def get_afk(self, user_id: int) -> Optional[tuple]:
        """Get AFK status"""
        cursor = self.conn.execute('SELECT reason, timestamp FROM afk_users WHERE user_id = ?', (user_id,))
        return cursor.fetchone()
    
    def is_afk(self, user_id: int) -> bool:
        """Check if user is AFK"""
        return self.get_afk(user_id) is not None
    
    def get_all_afk(self, guild_id: int) -> List[tuple]:
        """Get all AFK users in guild"""
        cursor = self.conn.execute('SELECT user_id, reason, timestamp FROM afk_users WHERE guild_id = ?', (guild_id,))
        return cursor.fetchall()
    
    def add_ignored_channel(self, channel_id: int, guild_id: int):
        """Add channel to ignore list"""
        self.conn.execute(
            'INSERT OR REPLACE INTO ignored_channels VALUES (?, ?)',
            (channel_id, guild_id)
        )
        self.conn.commit()
        self.ignored_channels.add(channel_id)
    
    def remove_ignored_channel(self, channel_id: int):
        """Remove channel from ignore list"""
        self.conn.execute('DELETE FROM ignored_channels WHERE channel_id = ?', (channel_id,))
        self.conn.commit()
        self.ignored_channels.discard(channel_id)
    
    def is_channel_ignored(self, channel_id: int) -> bool:
        """Check if channel is ignored"""
        return channel_id in self.ignored_channels
    
    def load_ignored_channels(self):
        """Load ignored channels from database"""
        cursor = self.conn.execute('SELECT channel_id FROM ignored_channels')
        self.ignored_channels = {row[0] for row in cursor.fetchall()}

class PostgreSQLManager:
    """🗄️ Async PostgreSQL Database Manager with Vector Database Integration"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.pool = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.database_url)
            self.initialized = True
            logger.info("✅ PostgreSQL pool initialized")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL initialization failed: {e}")
            return False
            
    async def get_connection(self):
        """Get PostgreSQL connection from pool"""
        if not self.initialized or not self.pool:
            return None
        return await self.pool.acquire()
    
    async def save_search(self, user_id: int, guild_id: int, search_type: str, query: str, results_count: int = 0):
        """Save search history to PostgreSQL"""
        if not self.initialized:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO search_history (user_id, guild_id, search_type, query, results_count, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                """, user_id, guild_id, search_type, query, results_count, datetime.now())
                logger.info(f"💾 Saved search: {search_type} by user {user_id}")
        except Exception as e:
            logger.error(f"❌ Error saving search: {e}")
    
    async def save_ensemble_conversation(self, user_id: int, guild_id: int, message: str, response: str, ai_provider: str, response_time_ms: int):
        """Save ensemble conversation to PostgreSQL with Vector Database integration"""
        if not self.initialized:
            return
        try:
            async with self.pool.acquire() as conn:
                # Save to PostgreSQL
                await conn.execute("""
                    INSERT INTO conversations (user_id, guild_id, message_content, bot_response, context_data, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, guild_id, message[:2000], response[:2000], 
                json.dumps({"ai_provider": ai_provider, "response_time_ms": response_time_ms}), datetime.now())
                
                # Also store in Vector Database for semantic search
                asyncio.create_task(memory_manager.store_conversation(
                    str(user_id), str(guild_id), message, response,
                    {"ai_provider": ai_provider, "response_time_ms": response_time_ms}
                ))
                
                logger.info(f"💾 Saved conversation: {ai_provider} for user {user_id}")
        except Exception as e:
            logger.error(f"❌ Error saving conversation: {e}")
    
    async def get_user_search_stats(self, user_id: int, guild_id: int):
        """Get user search statistics from PostgreSQL"""
        if not self.initialized:
            return {}
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT search_type, COUNT(*) as count
                    FROM search_history 
                    WHERE user_id = $1 AND guild_id = $2 
                    GROUP BY search_type
                """, user_id, guild_id)
                return {row['search_type']: row['count'] for row in results}
        except Exception as e:
            logger.error(f"❌ Error getting search stats: {e}")
            return {}
            
    async def get_conversation_history(self, user_id: int, guild_id: int, limit: int = 10):
        """Get conversation history with semantic search capabilities"""
        if not self.initialized:
            return []
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT message_content, bot_response, context_data, timestamp
                    FROM conversations 
                    WHERE user_id = $1 AND guild_id = $2 
                    ORDER BY timestamp DESC 
                    LIMIT $3
                """, user_id, guild_id, limit)
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Error getting conversation history: {e}")
            return []

class AffectionManager:
    """🌸 Manages affection system with hugs, kisses, marriages and ships 🌸"""
    
    def __init__(self):
        self.affection_db = "agua_affection.db"
        self.setup_database()
    
    def setup_database(self):
        """Setup affection tracking database"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        # User affection stats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS affection_stats (
                user_id INTEGER,
                guild_id INTEGER,
                hugs_given INTEGER DEFAULT 0,
                hugs_received INTEGER DEFAULT 0,
                kisses_given INTEGER DEFAULT 0,
                kisses_received INTEGER DEFAULT 0,
                pats_given INTEGER DEFAULT 0,
                pats_received INTEGER DEFAULT 0,
                cuddles_given INTEGER DEFAULT 0,
                cuddles_received INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, guild_id)
            )
        ''')
        
        # Add missing columns if they don't exist (for existing databases)
        for column in ['pats_given', 'pats_received', 'cuddles_given', 'cuddles_received']:
            try:
                cursor.execute(f'ALTER TABLE affection_stats ADD COLUMN {column} INTEGER DEFAULT 0')
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        # Marriages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marriages (
                user1_id INTEGER,
                user2_id INTEGER,
                guild_id INTEGER,
                married_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user1_id, user2_id, guild_id)
            )
        ''')
        
        # Ships/compatibility
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ships (
                user1_id INTEGER,
                user2_id INTEGER,
                guild_id INTEGER,
                compatibility INTEGER,
                shipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user1_id, user2_id, guild_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def update_affection(self, user_id: int, guild_id: int, action: str):
        """Update affection stats"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO affection_stats (user_id, guild_id) 
            VALUES (?, ?)
        ''', (user_id, guild_id))
        
        if action == "hug_given":
            cursor.execute('''
                UPDATE affection_stats 
                SET hugs_given = hugs_given + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "hug_received":
            cursor.execute('''
                UPDATE affection_stats 
                SET hugs_received = hugs_received + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "kiss_given":
            cursor.execute('''
                UPDATE affection_stats 
                SET kisses_given = kisses_given + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "kiss_received":
            cursor.execute('''
                UPDATE affection_stats 
                SET kisses_received = kisses_received + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "pat_given":
            cursor.execute('''
                UPDATE affection_stats 
                SET pats_given = pats_given + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "pat_received":
            cursor.execute('''
                UPDATE affection_stats 
                SET pats_received = pats_received + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "cuddle_given":
            cursor.execute('''
                UPDATE affection_stats 
                SET cuddles_given = cuddles_given + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        elif action == "cuddle_received":
            cursor.execute('''
                UPDATE affection_stats 
                SET cuddles_received = cuddles_received + 1 
                WHERE user_id = ? AND guild_id = ?
            ''', (user_id, guild_id))
        
        conn.commit()
        conn.close()
    
    def get_stats(self, user_id: int, guild_id: int) -> Dict:
        """Get user affection stats"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT hugs_given, hugs_received, kisses_given, kisses_received, pats_given, pats_received, cuddles_given, cuddles_received
            FROM affection_stats 
            WHERE user_id = ? AND guild_id = ?
        ''', (user_id, guild_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "hugs_given": result[0],
                "hugs_received": result[1], 
                "kisses_given": result[2],
                "kisses_received": result[3],
                "pats_given": result[4],
                "pats_received": result[5],
                "cuddles_given": result[6],
                "cuddles_received": result[7]
            }
        return {"hugs_given": 0, "hugs_received": 0, "kisses_given": 0, "kisses_received": 0, "pats_given": 0, "pats_received": 0, "cuddles_given": 0, "cuddles_received": 0}
    
    def create_marriage(self, user1_id: int, user2_id: int, guild_id: int) -> bool:
        """Create marriage between two users"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        try:
            # Ensure user1_id < user2_id for consistency
            if user1_id > user2_id:
                user1_id, user2_id = user2_id, user1_id
                
            cursor.execute('''
                INSERT INTO marriages (user1_id, user2_id, guild_id)
                VALUES (?, ?, ?)
            ''', (user1_id, user2_id, guild_id))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def remove_marriage(self, user1_id: int, user2_id: int, guild_id: int) -> bool:
        """Remove marriage"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        # Ensure user1_id < user2_id for consistency
        if user1_id > user2_id:
            user1_id, user2_id = user2_id, user1_id
            
        cursor.execute('''
            DELETE FROM marriages 
            WHERE user1_id = ? AND user2_id = ? AND guild_id = ?
        ''', (user1_id, user2_id, guild_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0
    
    def is_married(self, user1_id: int, user2_id: int, guild_id: int) -> bool:
        """Check if two users are married"""
        conn = sqlite3.connect(self.affection_db)
        cursor = conn.cursor()
        
        # Ensure user1_id < user2_id for consistency
        if user1_id > user2_id:
            user1_id, user2_id = user2_id, user1_id
            
        cursor.execute('''
            SELECT 1 FROM marriages 
            WHERE user1_id = ? AND user2_id = ? AND guild_id = ?
        ''', (user1_id, user2_id, guild_id))
        
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def calculate_ship_compatibility(self, user1_id: int, user2_id: int) -> int:
        """Calculate ship compatibility (1-100)"""
        # Use user IDs to generate consistent "random" compatibility
        combined = str(user1_id) + str(user2_id)
        seed = sum(ord(c) for c in combined)
        random.seed(seed)
        return random.randint(1, 100)

class PersonalityManager:
    """Manages adaptive personality for each user with automatic detection"""
    
    def __init__(self):
        self.personality_db = "agua_personality.db"
        self.setup_database()
    
    def setup_database(self):
        """Setup personality tracking database"""
        conn = sqlite3.connect(self.personality_db)
        cursor = conn.cursor()
        
        # User personality data - recreate with proper structure
        cursor.execute('DROP TABLE IF EXISTS user_personalities')
        cursor.execute('''
            CREATE TABLE user_personalities (
                user_id INTEGER PRIMARY KEY,
                personality TEXT DEFAULT 'normal',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User message interactions for personality detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                guild_id INTEGER,
                message TEXT,
                detected_style TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversation history for context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                guild_id INTEGER,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_personality_style(self, message: str) -> str:
        """Detect personality style from user message"""
        message = message.lower()
        
        # Waifu style detection
        waifu_patterns = ['uwu', 'owo', '~', '♡', 'nya', 'nyaa', 'kawaii', 'senpai', 'chan', 'kun', '>w<', 'nyan']
        if any(pattern in message for pattern in waifu_patterns):
            return 'waifu'
        
        # Maid style detection
        maid_patterns = ['goshujin-sama', 'sirvienta', 'master', 'amo', 'señor', 'señora', 'reverencia', 'humilde', 'servir']
        if any(pattern in message for pattern in maid_patterns):
            return 'maid'
        
        # Femboy style detection
        femboy_patterns = ['holi', 'holis', 'bestie', 'queen', 'king', 'slay', 'periodt', 'no cap']
        if any(pattern in message for pattern in femboy_patterns):
            return 'femboy'
        
        # Default to normal
        return 'normal'
    
    def save_user_message(self, user_id: int, guild_id: int, message: str):
        """Save user message and detect personality style"""
        try:
            detected_style = self.detect_personality_style(message)
            
            # Use timeout and WAL mode for better concurrency
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_interactions (user_id, guild_id, message, detected_style, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, guild_id, message, detected_style, datetime.now()))
            
            conn.commit()
            
            # Check if user has 10+ messages and calculate dominant personality
            cursor.execute('''
                SELECT COUNT(*) FROM user_interactions 
                WHERE user_id = ?
            ''', (user_id,))
            
            message_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Process personality calculation in separate connection to avoid locking
            if message_count >= 10:
                self.calculate_dominant_personality(user_id)
                
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when saving message for user {user_id}, skipping...")
                # Don't crash the bot, just log and continue
                return
            else:
                raise e
        except Exception as e:
            print(f"Error saving user message: {e}")
            # Ensure connection is closed even on error
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
    
    def calculate_dominant_personality(self, user_id: int):
        """Calculate and set dominant personality based on message history"""
        try:
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            # Get personality style counts from last 20 messages
            cursor.execute('''
                SELECT detected_style, COUNT(*) as count
                FROM (
                    SELECT detected_style
                    FROM user_interactions 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 20
                ) 
                GROUP BY detected_style
                ORDER BY count DESC
            ''', (user_id,))
            
            results = cursor.fetchall()
            
            if results:
                dominant_personality = results[0][0]
                conn.close()
                self.set_user_personality(user_id, dominant_personality)
            else:
                conn.close()
                
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when calculating personality for user {user_id}, skipping...")
                return
            else:
                raise e
        except Exception as e:
            print(f"Error calculating dominant personality: {e}")
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
    
    def get_user_personality(self, user_id: int) -> str:
        """Get user's personality type"""
        try:
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT personality FROM user_personalities WHERE user_id = ?",
                (user_id,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 'normal'
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when getting personality for user {user_id}, returning default...")
                return 'normal'
            else:
                raise e
        except Exception as e:
            print(f"Error getting user personality: {e}")
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
            return 'normal'
    
    def set_user_personality(self, user_id: int, personality: str):
        """Set user's personality type"""
        try:
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_personalities (user_id, personality, last_updated)
                VALUES (?, ?, ?)
            ''', (user_id, personality, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when setting personality for user {user_id}, skipping...")
                return
            else:
                raise e
        except Exception as e:
            print(f"Error setting user personality: {e}")
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
    
    def get_personality_response(self, personality: str, base_message: str, username: str) -> str:
        """Generate personality-adapted response"""
        if personality == 'waifu':
            return f"Nyaa~ ¡{username}-kun, qué alegría verte! {base_message} UwU 💖🐾"
        elif personality == 'maid':
            return f"Bienvenido a casa, {username}-sama. {base_message} 🫖✨"
        elif personality == 'femboy':
            return f"Holis~ {username}! {base_message} 🌈💅"
        else:
            return f"Hola, {username}. {base_message}"
    
    def get_interaction_response(self, personality: str, action: str, author: str, target: str, count: int) -> str:
        """Generate personality-adapted interaction response"""
        base_responses = {
            'hug': f"{author} le dio un fuerte abrazo a {target}",
            'kiss': f"{author} le dio un besito a {target}",
            'pat': f"{author} acarició suavemente a {target}",
            'poke': f"{author} tocó a {target}",
            'cuddle': f"{author} se acurrucó con {target}",
            'highfive': f"{author} chocó los cinco con {target}",
            'feed': f"{author} alimentó a {target}",
            'heal': f"{author} curó a {target}",
            'kick': f"{author} pateó a {target}",
            'punch': f"{author} golpeó a {target}",
            'slap': f"{author} abofeteó a {target}",
            'spray': f"{author} roció a {target}",
            'stare': f"{author} miró fijamente a {target}"
        }
        
        base_msg = base_responses.get(action, f"{author} hizo {action} a {target}")
        
        if personality == 'waifu':
            emoticons = ['(つ≧▽≦)つ', '(≧▽≦)', '(◕‿◕)', 'UwU', '>w<', '(´∀｀)♡']
            emoticon = random.choice(emoticons)
            return f"{base_msg}\n{target} lleva {count} en total {emoticon}"
        elif personality == 'maid':
            return f"{base_msg}\n{target} lleva {count} en total, goshujin-sama ✨"
        elif personality == 'femboy':
            return f"{base_msg}\n{target} lleva {count} en total bestie 💅✨"
        else:
            return f"{base_msg}\n{target} lleva {count} en total"
    
    def save_conversation(self, user_id: int, guild_id: int, message: str, response: str):
        """Save conversation for context"""
        try:
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversation_history (user_id, guild_id, message, response)
                VALUES (?, ?, ?, ?)
            ''', (user_id, guild_id, message, response))
            
            conn.commit()
            conn.close()
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when saving conversation for user {user_id}, skipping...")
                return
            else:
                raise e
        except Exception as e:
            print(f"Error saving conversation: {e}")
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
    
    def get_conversation_context(self, user_id: int, guild_id: int, limit: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        try:
            conn = sqlite3.connect(self.personality_db, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT message, response, timestamp
                FROM conversation_history 
                WHERE user_id = ? AND guild_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, guild_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{'message': r[0], 'response': r[1], 'timestamp': r[2]} for r in results]
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database locked when getting conversation context for user {user_id}, returning empty...")
                return []
            else:
                raise e
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
            return []

class GameManager:
    """Manages simple games"""
    
    def __init__(self):
        self.active_games = {}
        self.game_words = [
            'python', 'discord', 'kawaii', 'anime', 'manga', 'bot', 'code', 'game',
            'love', 'heart', 'smile', 'happy', 'friend', 'music', 'dance', 'sing'
        ]
    
    def create_hangman_game(self, user_id: int) -> Dict:
        """Create hangman game"""
        word = random.choice(self.game_words).upper()
        game = {
            'word': word,
            'guessed': set(),
            'wrong_guesses': 0,
            'max_wrong': 6,
            'display': ['_' if c.isalpha() else c for c in word]
        }
        self.active_games[f"hangman_{user_id}"] = game
        return game
    
    def guess_letter(self, user_id: int, letter: str) -> Dict:
        """Guess letter in hangman"""
        game_key = f"hangman_{user_id}"
        if game_key not in self.active_games:
            return None
        
        game = self.active_games[game_key]
        letter = letter.upper()
        
        if letter in game['guessed']:
            return {'status': 'already_guessed', 'game': game}
        
        game['guessed'].add(letter)
        
        if letter in game['word']:
            for i, c in enumerate(game['word']):
                if c == letter:
                    game['display'][i] = c
            
            if '_' not in game['display']:
                del self.active_games[game_key]
                return {'status': 'won', 'game': game}
            else:
                return {'status': 'correct', 'game': game}
        else:
            game['wrong_guesses'] += 1
            if game['wrong_guesses'] >= game['max_wrong']:
                del self.active_games[game_key]
                return {'status': 'lost', 'game': game}
            else:
                return {'status': 'wrong', 'game': game}

class ImageNavigationView(discord.ui.View):
    """Navigation view for image search results"""
    
    def __init__(self, images: List[Dict], query: str):
        super().__init__(timeout=300)
        self.images = images
        self.query = query
        self.current_index = 0
    
    @discord.ui.button(emoji="⬅️", style=discord.ButtonStyle.secondary)
    async def previous_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_index > 0:
            self.current_index -= 1
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.send_message("Ya estás en la primera imagen", ephemeral=True)
    
    @discord.ui.button(emoji="➡️", style=discord.ButtonStyle.secondary)
    async def next_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            embed = self.create_embed()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.send_message("Ya estás en la última imagen", ephemeral=True)
    
    @discord.ui.button(emoji="🔗", style=discord.ButtonStyle.success, label="Ver original")
    async def view_original(self, interaction: discord.Interaction, button: discord.ui.Button):
        image = self.images[self.current_index]
        embed = discord.Embed(
            title="🔗 Enlace original",
            description=f"[Abrir imagen original]({image['url']})",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
    
    def create_embed(self):
        image = self.images[self.current_index]
        embed = discord.Embed(
            title=f"🖼️ {self.query}",
            description=image.get('title', 'Sin título')[:200],
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        
        embed.set_image(url=image['url'])
        embed.add_field(name="📊 Imagen", value=f"{self.current_index + 1} de {len(self.images)}", inline=True)
        
        if image.get('context'):
            embed.add_field(name="🌐 Fuente", value=image['context'], inline=True)
        
        return embed

class ImageEditor:
    """Image editing functionality"""
    
    @staticmethod
    async def apply_filter(image_url: str, filter_type: str) -> BytesIO:
        """Apply filter to image"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        image = Image.open(BytesIO(image_data))
                        
                        if filter_type == 'blur':
                            image = image.filter(ImageFilter.BLUR)
                        elif filter_type == 'grayscale':
                            image = image.convert('L')
                        elif filter_type == 'sepia':
                            image = ImageEditor._apply_sepia(image)
                        elif filter_type == 'invert':
                            image = ImageEditor._invert_colors(image)
                        elif filter_type == 'pixelate':
                            image = ImageEditor._pixelate(image)
                        
                        output = BytesIO()
                        image.save(output, format='PNG')
                        output.seek(0)
                        return output
        except Exception as e:
            logger.error(f"Image editing error: {e}")
            return None
    
    @staticmethod
    def _apply_sepia(image):
        """Apply sepia filter"""
        pixels = image.load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                r, g, b = image.getpixel((i, j))[:3]
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[i, j] = (min(255, tr), min(255, tg), min(255, tb))
        return image
    
    @staticmethod
    def _invert_colors(image):
        """Invert image colors"""
        return Image.eval(image, lambda x: 255 - x)
    
    @staticmethod
    def _pixelate(image, pixel_size=10):
        """Pixelate image"""
        image = image.resize(
            (image.size[0] // pixel_size, image.size[1] // pixel_size),
            Image.NEAREST
        )
        image = image.resize(
            (image.size[0] * pixel_size, image.size[1] * pixel_size),
            Image.NEAREST
        )
        return image

class AutoModManager:
    """Sistema de AutoMod completo con filtrado automático y configuración persistente"""
    
    def __init__(self):
        self.automod_db = "moderation.db"
        self.setup_database()
        self.server_configs = {}
        self.load_all_configs()
    
    def setup_database(self):
        """Configurar base de datos de AutoMod"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        # Tabla de configuración por servidor
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS automod_config (
                guild_id INTEGER PRIMARY KEY,
                filtered_words TEXT DEFAULT '[]',
                action_type TEXT DEFAULT 'delete',
                enabled INTEGER DEFAULT 1,
                log_channel_id INTEGER DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de infracciones
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS automod_infractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                user_id INTEGER,
                channel_id INTEGER,
                message_content TEXT,
                detected_words TEXT,
                action_taken TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de reglas de AutoMod nativo (si está disponible)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS native_automod_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                rule_id INTEGER,
                rule_name TEXT,
                trigger_type TEXT,
                keywords TEXT,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_all_configs(self):
        """Cargar todas las configuraciones de servidores"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT guild_id, filtered_words, action_type, enabled, log_channel_id FROM automod_config")
        results = cursor.fetchall()
        
        for row in results:
            guild_id, filtered_words, action_type, enabled, log_channel_id = row
            self.server_configs[guild_id] = {
                'filtered_words': json.loads(filtered_words),
                'action_type': action_type,
                'enabled': bool(enabled),
                'log_channel_id': log_channel_id
            }
        
        conn.close()
        logger.info(f"Cargadas configuraciones de AutoMod para {len(self.server_configs)} servidores")
    
    def save_config(self, guild_id: int, config: dict):
        """Guardar configuración de servidor"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO automod_config 
            (guild_id, filtered_words, action_type, enabled, log_channel_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            guild_id,
            json.dumps(config['filtered_words']),
            config['action_type'],
            int(config['enabled']),
            config.get('log_channel_id'),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        # Actualizar cache
        self.server_configs[guild_id] = config
    
    def get_config(self, guild_id: int) -> dict:
        """Obtener configuración de servidor"""
        return self.server_configs.get(guild_id, {
            'filtered_words': [],
            'action_type': 'delete',
            'enabled': True,
            'log_channel_id': None
        })
    
    def add_filtered_words(self, guild_id: int, words: list) -> int:
        """Agregar palabras al filtro"""
        config = self.get_config(guild_id)
        existing_words = set(config['filtered_words'])
        new_words = [word.lower().strip() for word in words if word.lower().strip() not in existing_words]
        
        config['filtered_words'].extend(new_words)
        self.save_config(guild_id, config)
        
        return len(new_words)
    
    def remove_filtered_words(self, guild_id: int, words: list) -> int:
        """Eliminar palabras del filtro"""
        config = self.get_config(guild_id)
        words_to_remove = [word.lower().strip() for word in words]
        removed_count = 0
        
        for word in words_to_remove:
            if word in config['filtered_words']:
                config['filtered_words'].remove(word)
                removed_count += 1
        
        self.save_config(guild_id, config)
        return removed_count
    
    def clear_filtered_words(self, guild_id: int):
        """Limpiar todas las palabras filtradas"""
        config = self.get_config(guild_id)
        config['filtered_words'] = []
        self.save_config(guild_id, config)
    
    def set_action_type(self, guild_id: int, action_type: str):
        """Configurar tipo de acción"""
        config = self.get_config(guild_id)
        config['action_type'] = action_type
        self.save_config(guild_id, config)
    
    def set_log_channel(self, guild_id: int, channel_id: int):
        """Configurar canal de logs"""
        config = self.get_config(guild_id)
        config['log_channel_id'] = channel_id
        self.save_config(guild_id, config)
    
    def check_message(self, guild_id: int, message_content: str) -> dict:
        """Verificar si un mensaje contiene palabras filtradas"""
        config = self.get_config(guild_id)
        
        if not config['enabled'] or not config['filtered_words']:
            return {'detected': False}
        
        content_lower = message_content.lower()
        detected_words = []
        
        for word in config['filtered_words']:
            if word.lower() in content_lower:
                detected_words.append(word)
        
        if detected_words:
            return {
                'detected': True,
                'words': detected_words,
                'action': config['action_type']
            }
        
        return {'detected': False}
    
    def log_infraction(self, guild_id: int, user_id: int, channel_id: int, 
                      message_content: str, detected_words: list, action_taken: str):
        """Registrar infracción"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO automod_infractions 
            (guild_id, user_id, channel_id, message_content, detected_words, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (guild_id, user_id, channel_id, message_content, 
              json.dumps(detected_words), action_taken))
        
        conn.commit()
        conn.close()
    
    def get_infractions_stats(self, guild_id: int, days: int = 7) -> dict:
        """Obtener estadísticas de infracciones"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        # Infracciones en los últimos días
        cursor.execute('''
            SELECT COUNT(*) FROM automod_infractions 
            WHERE guild_id = ? AND timestamp > datetime('now', '-{} days')
        '''.format(days), (guild_id,))
        
        recent_count = cursor.fetchone()[0]
        
        # Usuario con más infracciones
        cursor.execute('''
            SELECT user_id, COUNT(*) as count FROM automod_infractions 
            WHERE guild_id = ? AND timestamp > datetime('now', '-{} days')
            GROUP BY user_id ORDER BY count DESC LIMIT 1
        '''.format(days), (guild_id,))
        
        top_user = cursor.fetchone()
        
        # Palabras más detectadas
        cursor.execute('''
            SELECT detected_words FROM automod_infractions 
            WHERE guild_id = ? AND timestamp > datetime('now', '-{} days')
        '''.format(days), (guild_id,))
        
        all_words = []
        for row in cursor.fetchall():
            words = json.loads(row[0])
            all_words.extend(words)
        
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        conn.close()
        
        return {
            'recent_count': recent_count,
            'top_user': top_user,
            'top_words': top_words
        }
    
    def save_native_rule(self, guild_id: int, rule_id: int, rule_name: str, 
                        trigger_type: str, keywords: list, created_by: int):
        """Guardar regla de AutoMod nativo"""
        conn = sqlite3.connect(self.automod_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO native_automod_rules 
            (guild_id, rule_id, rule_name, trigger_type, keywords, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (guild_id, rule_id, rule_name, trigger_type, json.dumps(keywords), created_by))
        
        conn.commit()
        conn.close()

class NekotinaBot(commands.Bot):
    """Main bot class inspired by Nekotina"""
    
    def __init__(self):
        # Intents optimizados para DM support completo
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        intents.dm_messages = True  # CRÍTICO para comandos DM
        intents.reactions = True
        intents.typing = True
        intents.members = True  # Para user profiles en DM
        intents.presences = True  # Para status en DM
        
        # Habilitar intents de AutoMod si están disponibles
        try:
            intents.auto_moderation_configuration = True
            intents.auto_moderation_execution = True
            logger.info("Intents de AutoMod habilitados correctamente")
        except AttributeError:
            logger.info("Intents de AutoMod no disponibles en esta versión de discord.py")
        
        # Get application ID from environment or bot token
        application_id = os.getenv('APPLICATION_ID')
        if application_id:
            try:
                application_id = int(application_id)
            except ValueError:
                logger.warning("APPLICATION_ID no es un número válido, usando None")
                application_id = None
        
        super().__init__(
            command_prefix='$',  # Prefix para comandos tradicionales
            intents=intents,
            help_command=None,
            case_insensitive=True,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False),
            application_id=application_id
        )
        
        self.gif_provider = GifProvider()
        self.afk_manager = AFKManager()
        self.affection_manager = AffectionManager()
        self.personality_manager = PersonalityManager()
        self.game_manager = GameManager()
        self.ai_provider = AIProvider()
        self.music_provider = MusicProvider()
        self.web_scraper = WebScraper()
        self.search_provider = AdvancedSearchProvider()
        self.automod_manager = AutoModManager()
        
        # Initialize multimodal system
        self._multimodal_initialized = False
        
        # User cooldown system for AI interactions 
        self.user_cooldowns = {}  # {user_id: last_interaction_time}
        self.ai_cooldown_duration = 30  # 30 seconds cooldown between AI interactions
        
        # User cooldown system for social interactions (pat, hug, kiss, etc.)

        
        # Load ignored channels
        self.afk_manager.load_ignored_channels()
        
        # Fun facts for commands
        self.cat_facts = [
            "Los gatos pueden hacer más de 100 sonidos vocales diferentes.",
            "Los gatos duermen entre 12-16 horas al día.",
            "Los gatos tienen un tercer párpado llamado membrana nictitante.",
            "Un grupo de gatos se llama 'clowder'.",
            "Los gatos pueden rotar sus orejas 180 grados."
        ]
        
        self.dog_facts = [
            "Los perros tienen aproximadamente 300 millones de receptores olfativos.",
            "Los perros pueden aprender más de 150 palabras.",
            "Los perros sudan a través de sus patas.",
            "El perro promedio puede correr a 19 mph.",
            "Los perros tienen tres párpados."
        ]
    
    def extract_important_data(self, message: str) -> Dict[str, str]:
        """Extract important data from user message for memory storage"""
        data = {}
        
        # Extract names
        name_patterns = [
            r"me llamo (\w+)",
            r"mi nombre es (\w+)",
            r"soy (\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message.lower())
            if match:
                data["nombre"] = match.group(1)
        
        # Extract dates
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\d{1,2}-\d{1,2}-\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, message)
            if match:
                data["fecha_mencionada"] = match.group(1)
        
        # Extract preferences
        if "me gusta" in message.lower():
            preference = re.search(r"me gusta (.+)", message.lower())
            if preference:
                data["preferencia"] = preference.group(1)
        
        # Extract mood/emotions
        emotions = {
            "feliz": ["feliz", "contento", "alegre", "bien"],
            "triste": ["triste", "mal", "deprimido", "down"],
            "enojado": ["enojado", "molesto", "furioso", "angry"]
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in message.lower() for keyword in keywords):
                data["estado_animo"] = emotion
                break
        
        return data

    async def process_special_triggers(self, message):
        """DESACTIVADO - Process special triggers: $, @SakuraBot, and Sakura commands with command integration"""
        # SISTEMA DESACTIVADO PARA EVITAR DUPLICACIONES
        # Ahora se usa _handle_ai_response en su lugar
        return False
    
    async def process_dollar_command(self, message, command_text):
        """Process $ commands by mapping them to slash commands"""
        try:
            parts = command_text.split()
            if not parts:
                return False
                
            command_name = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Map common commands
            command_mapping = {
                'youtube': self.execute_youtube_search,
                'search': self.execute_youtube_search,
                'ytsearch': self.execute_youtube_search,
                'images': self.execute_image_search,
                'img': self.execute_image_search,
                'ai': self.execute_ai_command,
                'generar_imagen': self.execute_image_generation,
                'gen': self.execute_image_generation,
                'translate': self.execute_translate,
                'weather': self.execute_weather,
                'ping': self.execute_ping,
                'help': self.execute_help,
                'tts': self.execute_tts,
                'act': self.execute_act,
                'interact': self.execute_interact,
                'bonk': self.execute_bonk,
                'marry': self.execute_marry,
                'divorce': self.execute_divorce,
                'userinfo': self.execute_userinfo,
                'serverinfo': self.execute_serverinfo,
                'roll': self.execute_roll,
                'rps': self.execute_rps,
                'lucky': self.execute_lucky,
                'catfact': self.execute_catfact,
                'dogfact': self.execute_dogfact,
                'joke': self.execute_joke,
                'meme': self.execute_meme,
                'quote': self.execute_quote,
                '8ball': self.execute_8ball,
                'ship': self.execute_ship,
                'afk': self.execute_afk,
                'blur': self.execute_blur,
                'avatar': self.execute_avatar,
                'qr': self.execute_qr,
                'math': self.execute_math,
                'password': self.execute_password,
                'stats': self.execute_stats
            }
            
            if command_name in command_mapping:
                await command_mapping[command_name](message, args)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error processing dollar command: {e}")
            await message.channel.send(f"Error ejecutando comando: {str(e)}")
            return True
    
    # Command execution methods
    async def execute_youtube_search(self, message, args):
        """Execute YouTube search command"""
        if not args:
            await message.channel.send("Uso: $ youtube <término de búsqueda>")
            return
        query = " ".join(args)
        results = await self.search_provider.search_youtube(query, 5)
        await message.channel.send(f"🎵 **Resultados de YouTube para: {query}**\n{results}")
    
    async def execute_image_search(self, message, args):
        """Execute image search command"""
        if not args:
            await message.channel.send("Uso: $ images <término de búsqueda>")
            return
        query = " ".join(args)
        results = await self.search_provider.search_images(query, 5)
        await message.channel.send(f"🖼️ **Imágenes para: {query}**\n{results}")
    
    async def execute_ai_command(self, message, args):
        """Execute AI command"""
        if len(args) < 2:
            await message.channel.send("Uso: $ ai <modelo> <mensaje>")
            return
        model = args[0]
        prompt = " ".join(args[1:])
        response = await self.ai_provider.get_ai_response(model, prompt, message.author.id)
        await message.channel.send(response)
    
    async def execute_image_generation(self, message, args):
        """Execute image generation command"""
        if not args:
            await message.channel.send("Uso: $ generar_imagen <descripción>")
            return
        prompt = " ".join(args)
        try:
            image_data = await self.ai_provider.generate_image_sdxl(prompt)
            if image_data:
                await message.channel.send(file=discord.File(image_data, "generated_image.png"))
            else:
                await message.channel.send("Error generando imagen")
        except Exception as e:
            await message.channel.send(f"Error: {str(e)}")
    
    async def execute_translate(self, message, args):
        """Execute translate command"""
        if len(args) < 3:
            await message.channel.send("Uso: $ translate <idioma_origen> <idioma_destino> <texto>")
            return
        # Implementation for translate
        await message.channel.send("Comando de traducción no implementado completamente")
    
    async def execute_weather(self, message, args):
        """Execute weather command"""
        if not args:
            await message.channel.send("Uso: $ weather <ciudad>")
            return
        # Implementation for weather
        await message.channel.send("Comando de clima no implementado completamente")
    
    async def execute_ping(self, message, args):
        """Execute ping command"""
        latency = round(self.latency * 1000)
        await message.channel.send(f"🏓 Pong! Latencia: {latency}ms")
    
    async def execute_help(self, message, args):
        """Execute help command"""
        help_text = """
**📚 Comandos disponibles con $:**
`$ youtube <búsqueda>` - Buscar videos
`$ images <búsqueda>` - Buscar imágenes  
`$ ai <modelo> <mensaje>` - Chat con IA
`$ generar_imagen <descripción>` - Generar imagen
`$ ping` - Ver latencia
`$ roll <dados>` - Lanzar dados
`$ act <acción>` - Realizar acción
`$ interact <acción> @usuario` - Interactuar
`$ lucky` - Probar suerte
`$ catfact` - Dato de gatos
`$ dogfact` - Dato de perros
`$ joke` - Chiste
`$ meme` - Meme aleatorio
`$ quote` - Cita inspiracional
`$ 8ball <pregunta>` - Bola 8 mágica
`$ ship @usuario1 @usuario2` - Medidor de amor
`$ afk [motivo]` - Establecer AFK
"""
        await message.channel.send(help_text)
    
    async def execute_tts(self, message, args):
        """Execute TTS command"""
        if not args:
            await message.channel.send("Uso: $ tts <texto>")
            return
        # Implementation for TTS
        await message.channel.send("Comando TTS no implementado completamente")
    
    async def execute_act(self, message, args):
        """Execute act command"""
        if not args:
            await message.channel.send("Uso: $ act <acción>")
            return
        action = args[0].lower()
        if action in self.gif_provider.act_actions:
            gif_url = await self.gif_provider.get_gif(action)
            description = self.gif_provider.action_descriptions.get(action, action)
            if gif_url:
                embed = discord.Embed(description=f"**{message.author.mention} {description}**", color=COLORES_KAWAII["ROSA_PASTEL"])
                embed.set_image(url=gif_url)
                await message.channel.send(embed=embed)
            else:
                await message.channel.send(f"**{message.author.mention} {description}**")
        else:
            await message.channel.send(f"Acción '{action}' no disponible")
    
    async def execute_interact(self, message, args):
        """Execute interact command"""
        if len(args) < 2:
            await message.channel.send("Uso: $ interact <acción> @usuario")
            return
        # Implementation for interact
        await message.channel.send("Comando interact no implementado completamente")
    
    async def execute_bonk(self, message, args):
        """Execute bonk command"""
        if not message.mentions:
            await message.channel.send("Uso: $ bonk @usuario")
            return
        target = message.mentions[0]
        await message.channel.send(f"**{message.author.mention} le da un bonk kawaii a {target.mention}** 🔨✨")
    
    async def execute_marry(self, message, args):
        """Execute marry command"""
        if not message.mentions:
            await message.channel.send("Uso: $ marry @usuario")
            return
        # Implementation for marry
        await message.channel.send("Comando marry no implementado completamente")
    
    async def execute_divorce(self, message, args):
        """Execute divorce command"""
        # Implementation for divorce
        await message.channel.send("Comando divorce no implementado completamente")
    
    async def execute_userinfo(self, message, args):
        """Execute userinfo command"""
        target = message.mentions[0] if message.mentions else message.author
        embed = discord.Embed(title=f"👤 Información de {target.display_name}", color=target.color)
        embed.set_thumbnail(url=target.display_avatar.url)
        embed.add_field(name="🏷️ Nombre", value=target.name, inline=True)
        embed.add_field(name="🆔 ID", value=target.id, inline=True)
        embed.add_field(name="📅 Cuenta creada", value=target.created_at.strftime("%d/%m/%Y"), inline=True)
        await message.channel.send(embed=embed)
    
    async def execute_serverinfo(self, message, args):
        """Execute serverinfo command"""
        guild = message.guild
        if not guild:
            await message.channel.send("Este comando solo funciona en servidores")
            return
        embed = discord.Embed(title=f"📊 Información de {guild.name}", color=COLORES_KAWAII["LAVANDA_KAWAII"])
        embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
        embed.add_field(name="👑 Propietario", value=guild.owner.mention if guild.owner else "Desconocido", inline=True)
        embed.add_field(name="👥 Miembros", value=guild.member_count, inline=True)
        embed.add_field(name="📅 Creado", value=guild.created_at.strftime("%d/%m/%Y"), inline=True)
        await message.channel.send(embed=embed)
    
    async def execute_roll(self, message, args):
        """Execute roll command"""
        if not args:
            sides = 6
            count = 1
        else:
            try:
                if 'd' in args[0]:
                    count, sides = map(int, args[0].split('d'))
                else:
                    sides = int(args[0])
                    count = 1
            except:
                await message.channel.send("Formato inválido. Usa: $ roll 6 o $ roll 2d6")
                return
        
        results = [random.randint(1, sides) for _ in range(count)]
        total = sum(results)
        
        if count == 1:
            await message.channel.send(f"🎲 **{message.author.mention}** lanzó un dado de {sides} caras: **{total}**")
        else:
            await message.channel.send(f"🎲 **{message.author.mention}** lanzó {count} dados de {sides} caras: {results} = **{total}**")
    
    async def execute_rps(self, message, args):
        """Execute rock paper scissors command"""
        if not args:
            await message.channel.send("Uso: $ rps <piedra/papel/tijeras>")
            return
        
        choices = ['piedra', 'papel', 'tijeras']
        user_choice = args[0].lower()
        
        if user_choice not in choices:
            await message.channel.send("Elige: piedra, papel o tijeras")
            return
        
        bot_choice = random.choice(choices)
        
        if user_choice == bot_choice:
            result = "¡Empate!"
        elif (user_choice == 'piedra' and bot_choice == 'tijeras') or \
             (user_choice == 'papel' and bot_choice == 'piedra') or \
             (user_choice == 'tijeras' and bot_choice == 'papel'):
            result = "¡Ganaste!"
        else:
            result = "¡Perdiste!"
        
        await message.channel.send(f"Tu elección: **{user_choice}**\nMi elección: **{bot_choice}**\n{result}")
    
    async def execute_lucky(self, message, args):
        """Execute lucky command"""
        luck = random.randint(1, 100)
        if luck >= 90:
            message_text = "¡Increíblemente afortunado! ✨"
        elif luck >= 70:
            message_text = "¡Muy afortunado! 🍀"
        elif luck >= 50:
            message_text = "Suerte promedio 😊"
        elif luck >= 30:
            message_text = "Un poco de mala suerte 😅"
        else:
            message_text = "¡Cuidado hoy! 😰"
        
        await message.channel.send(f"🍀 **{message.author.mention}** tu suerte del día: **{luck}%** - {message_text}")
    
    async def execute_catfact(self, message, args):
        """Execute cat fact command"""
        fact = random.choice(self.cat_facts)
        await message.channel.send(f"🐱 **Dato Curioso sobre Gatos:**\n{fact}")
    
    async def execute_dogfact(self, message, args):
        """Execute dog fact command"""
        fact = random.choice(self.dog_facts)
        await message.channel.send(f"🐶 **Dato Curioso sobre Perros:**\n{fact}")
    
    async def execute_joke(self, message, args):
        """Execute joke command"""
        jokes = [
            "¿Por qué los pájaros vuelan hacia el sur en invierno? Porque es demasiado lejos para caminar.",
            "¿Qué le dice un iguana a su hermana gemela? Somos iguanitas.",
            "¿Cómo se llama el campeón de buceo japonés? Tokofondo.",
            "¿Por qué las focas del Pacífico miran hacia abajo? Porque no pueden ver hacia arriba.",
            "¿Qué hace una abeja en el gimnasio? ¡Zum-ba!"
        ]
        joke = random.choice(jokes)
        await message.channel.send(f"😄 **Chiste Kawaii:**\n{joke}")
    
    async def execute_meme(self, message, args):
        """Execute meme command"""
        await message.channel.send("🎭 ¡Aquí tienes un meme! (función de memes no implementada completamente)")
    
    async def execute_quote(self, message, args):
        """Execute quote command"""
        quotes = [
            "El éxito es ir de fracaso en fracaso sin perder el entusiasmo. - Winston Churchill",
            "La vida es lo que pasa mientras estás ocupado haciendo otros planes. - John Lennon",
            "El futuro pertenece a quienes creen en la belleza de sus sueños. - Eleanor Roosevelt",
            "No es la especie más fuerte la que sobrevive, sino la más adaptable al cambio. - Charles Darwin",
            "La felicidad no es algo hecho. Viene de tus propias acciones. - Dalai Lama"
        ]
        quote = random.choice(quotes)
        await message.channel.send(f"💭 **Cita Inspiracional:**\n*{quote}*")
    
    async def execute_8ball(self, message, args):
        """Execute 8ball command"""
        if not args:
            await message.channel.send("Uso: $ 8ball <pregunta>")
            return
        
        responses = [
            "Sí, definitivamente", "Es cierto", "Sin duda", "Sí", "Puedes confiar en ello",
            "Como yo lo veo, sí", "Muy probable", "Las perspectivas son buenas", "Sí",
            "Las señales apuntan a que sí", "Respuesta confusa, intenta de nuevo",
            "Pregunta de nuevo más tarde", "Mejor no te lo digo ahora",
            "No puedo predecirlo ahora", "Concéntrate y pregunta de nuevo",
            "No cuentes con ello", "Mi respuesta es no", "Mis fuentes dicen que no",
            "Las perspectivas no son tan buenas", "Muy dudoso"
        ]
        
        response = random.choice(responses)
        question = " ".join(args)
        await message.channel.send(f"🎱 **Pregunta:** {question}\n**Respuesta:** {response}")
    
    async def execute_ship(self, message, args):
        """Execute ship command"""
        if len(message.mentions) < 2:
            await message.channel.send("Uso: $ ship @usuario1 @usuario2")
            return
        
        user1 = message.mentions[0]
        user2 = message.mentions[1]
        compatibility = random.randint(0, 100)
        
        if compatibility >= 90:
            msg = "¡Perfecta pareja! 💕"
        elif compatibility >= 70:
            msg = "¡Muy compatibles! 💖"
        elif compatibility >= 50:
            msg = "Buena química 💘"
        elif compatibility >= 30:
            msg = "Pueden intentarlo 💙"
        else:
            msg = "Mejor como amigos 💙"
        
        await message.channel.send(f"💕 **Ship Meter**\n{user1.mention} 💖 {user2.mention}\n**Compatibilidad: {compatibility}%**\n{msg}")
    
    async def execute_afk(self, message, args):
        """Execute AFK command"""
        reason = " ".join(args) if args else "AFK"
        self.afk_manager.set_afk(message.author.id, reason)
        await message.channel.send(f"😴 **{message.author.mention}** está ahora AFK: {reason}")
    
    async def execute_blur(self, message, args):
        """Execute blur command"""
        await message.channel.send("🖼️ Comando blur no implementado completamente")
    
    async def execute_avatar(self, message, args):
        """Execute avatar command"""
        target = message.mentions[0] if message.mentions else message.author
        embed = discord.Embed(title=f"🖼️ Avatar de {target.display_name}", color=target.color)
        embed.set_image(url=target.display_avatar.url)
        await message.channel.send(embed=embed)
    
    async def execute_qr(self, message, args):
        """Execute QR command"""
        if not args:
            await message.channel.send("Uso: $ qr <texto>")
            return
        await message.channel.send("📱 Comando QR no implementado completamente")
    
    async def execute_math(self, message, args):
        """Execute math command"""
        if not args:
            await message.channel.send("Uso: $ math <expresión>")
            return
        
        try:
            expression = " ".join(args)
            # Simple math evaluation (be careful with eval!)
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max})
            await message.channel.send(f"🧮 **Resultado:** `{expression}` = **{result}**")
        except:
            await message.channel.send("❌ Expresión matemática inválida")
    
    async def execute_password(self, message, args):
        """Execute password command"""
        length = 12
        if args and args[0].isdigit():
            length = min(int(args[0]), 50)
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        password = ''.join(random.choice(chars) for _ in range(length))
        
        try:
            await message.author.send(f"🔐 **Contraseña generada:** `{password}`")
            await message.channel.send("✅ Contraseña enviada por mensaje privado")
        except:
            await message.channel.send("❌ No pude enviarte un mensaje privado")
    
    async def execute_stats(self, message, args):
        """Execute stats command"""
        embed = discord.Embed(title="📊 Estadísticas del Bot", color=COLORES_KAWAII["LAVANDA_KAWAII"])
        embed.add_field(name="🏓 Latencia", value=f"{round(self.latency * 1000)}ms", inline=True)
        embed.add_field(name="🖥️ Servidores", value=len(self.guilds), inline=True)
        embed.add_field(name="👥 Usuarios", value=len(self.users), inline=True)
        await message.channel.send(embed=embed)
    
    async def connect_lavalink_nodes(self):
        """Conectar a nodos Lavalink con sistema de fallback kawaii"""
        for attempt in range(self.max_connection_attempts):
            if self.current_node_index >= len(NODOS_LAVALINK):
                self.current_node_index = 0
            
            nodo_config = NODOS_LAVALINK[self.current_node_index]
            
            try:
                logger.info(f"🔄 Intentando conectar a {nodo_config['identifier']} ({nodo_config['region']})")
                
                # Crear nodo Wavelink con URI construida
                uri = f"{'wss' if nodo_config.get('secure', True) else 'ws'}://{nodo_config['host']}:{nodo_config['port']}"
                node = wavelink.Node(
                    uri=uri,
                    password=nodo_config['password'],
                    identifier=nodo_config['identifier']
                )
                
                # Conectar al pool de nodos (Wavelink 3.x API)
                await wavelink.Pool.connect(nodes=[node], client=self)
                
                # Log de éxito
                timestamp = datetime.now().strftime("%H:%M:%S")
                success_log = {
                    "timestamp": timestamp,
                    "node": nodo_config['identifier'],
                    "region": nodo_config['region'],
                    "status": "CONNECTED",
                    "attempt": attempt + 1
                }
                self.connection_logs.append(success_log)
                self.node_status[nodo_config['identifier']] = "CONNECTED"
                self.connected_nodes.append(nodo_config)
                
                logger.info(f"✅ [{nodo_config['identifier']}] CONECTADO exitosamente - {nodo_config['region']}")
                
                # Enviar mensaje kawaii si hay canal disponible (inicializar si no existe)
                if not hasattr(self, 'notification_channel'):
                    self.notification_channel = None
                    
                if self.notification_channel:
                    embed = discord.Embed(
                        title="🎶 ¡Conexión Musical Exitosa! UwU",
                        description=f"¡Kyaa~! Me conecté exitosamente a **{nodo_config['identifier']}** 🌸\n"
                                  f"**Región:** {nodo_config['region']}\n"
                                  f"¡Ya puedo reproducir música para ti! (>w<)",
                        color=COLORES_KAWAII["ROSA_PASTEL"],
                        timestamp=datetime.now()
                    )
                    await self.notification_channel.send(embed=embed)
                
                return True
                
            except Exception as e:
                # Log de error
                timestamp = datetime.now().strftime("%H:%M:%S")
                error_log = {
                    "timestamp": timestamp,
                    "node": nodo_config['identifier'],
                    "region": nodo_config['region'],
                    "status": "FAILED",
                    "error": str(e),
                    "attempt": attempt + 1
                }
                self.connection_logs.append(error_log)
                self.node_status[nodo_config['identifier']] = "FAILED"
                
                logger.warning(f"❌ [{nodo_config['identifier']}] FALLÓ - {nodo_config['region']}: {e}")
                
                # Avanzar al siguiente nodo
                self.current_node_index += 1
                await asyncio.sleep(5)  # Esperar 5 segundos antes del siguiente intento
        
        # Si llegamos aquí, no se pudo conectar a ningún nodo
        logger.error("❌ No se pudo conectar a ningún nodo Lavalink")
        return False

    async def setup_hook(self):
        """OPTIMIZED Setup - WITH AUTOMATIC SLASH SYNC ENABLED"""
        try:
            log_kawaii_info("Starting Sakura IA with kawaii slash commands enabled! ♡")
            
            # Initialize managers with sync enabled
            self.rate_limiter = RateLimitManager()
            self.sync_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=300.0, max_timeout=3600.0)
            
            # Enable automatic slash command sync
            self.commands_synced = False  # Allow sync to happen
            self.sync_in_progress = False
            self.last_sync_attempt = 0
            self.sync_attempts = 0
            self.max_sync_attempts = 3  # Allow multiple sync attempts
            self.emergency_mode = False
            self.prefix_mode_active = False  # Allow slash commands
            
            # Inicializar sistema de música con nodos Lavalink
            self.current_node_index = 0
            self.connection_attempts = 0
            self.max_connection_attempts = 10
            self.reconnect_task = None
            self.node_status = {}
            self.connected_nodes = []
            self.music_sessions = 0
            self.songs_played = 0
            self.connection_logs = []
            
            # Intentar conectar a nodos Lavalink con protección
            log_kawaii_info("Starting music system connection... *plays cute melody*")
            await self.connect_lavalink_nodes_safe()
            
            # ATTEMPT SLASH COMMAND SYNC
            log_kawaii_info("Syncing kawaii slash commands... *sparkles*")
            sync_success = await self._attempt_resilient_sync()
            
            if sync_success:
                log_kawaii_success("Slash commands synced perfectly!")
            else:
                log_kawaii_warning("Slash sync had issues but we'll keep trying!")
            
            # Initialize Redis Cache System
            log_kawaii_info("Setting up Redis cache for fast responses...")
            self.redis_manager = redis_manager
            asyncio.create_task(self._setup_redis())
            
            # Initialize PostgreSQL Database
            log_kawaii_info("Connecting to PostgreSQL database...")
            asyncio.create_task(self._setup_postgresql())
            
            # Initialize Pinecone Vector Memory
            log_kawaii_info("Initializing AI memory system...")
            asyncio.create_task(self._setup_pinecone())
            
            # Initialize providers
            log_kawaii_success("AI ensemble system ready")
            log_kawaii_success("Search and multimedia systems active")
            
        except Exception as e:
            logger.error(f"Setup error: {e}")
            logger.info("⚠️ Bot will continue with limited functionality")
    
    async def _setup_redis(self):
        """Initialize Redis connection and setup"""
        try:
            await self.redis_manager.connect()
            if self.redis_manager.connected:
                logger.info("✅ Redis Cache System connected and ready")
                # Initialize bot statistics in Redis
                await self.redis_manager.set("bot:startup", datetime.utcnow().isoformat(), ttl=86400)
            else:
                logger.warning("⚠️ Redis connection failed - continuing without cache")
        except Exception as e:
            logger.error(f"Redis setup error: {e}")
            logger.info("⚠️ Bot will continue without Redis cache")
    
    async def _setup_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            await self.postgresql_manager.initialize()
            if self.postgresql_manager.initialized:
                logger.info("✅ PostgreSQL Database connected and ready")
                # Create missing tables
                await self._create_postgresql_tables()
            else:
                logger.warning("⚠️ PostgreSQL connection failed - using fallback storage")
        except Exception as e:
            logger.error(f"PostgreSQL setup error: {e}")
            logger.info("⚠️ Bot will continue without PostgreSQL")
    
    async def _setup_pinecone(self):
        """Initialize Pinecone Vector Database"""
        try:
            await memory_manager.initialize()
            if memory_manager.initialized:
                logger.info("✅ Pinecone Vector Memory System connected and ready")
                self.memory_manager = memory_manager
            else:
                logger.warning("⚠️ Pinecone connection failed - AI memory features disabled")
        except Exception as e:
            logger.error(f"Pinecone setup error: {e}")
            logger.info("⚠️ Bot will continue without vector memory")
    
    async def _create_postgresql_tables(self):
        """Create necessary PostgreSQL tables"""
        if not self.postgresql_manager.initialized:
            return
        try:
            async with self.postgresql_manager.pool.acquire() as conn:
                # Create search history table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id SERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        guild_id BIGINT NOT NULL,
                        search_type VARCHAR(50) NOT NULL,
                        query TEXT NOT NULL,
                        results_count INTEGER DEFAULT 0,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                logger.info("✅ PostgreSQL tables verified/created")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
    
    async def _attempt_resilient_sync(self) -> bool:
        """Attempt ONE protected command sync with full circuit breaker protection"""
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
        
        # Check cooldown (prevent rapid attempts) - Reduced for manual admin activation
        current_time = time.time()
        min_cooldown = 120 if hasattr(self, '_admin_requested_sync') else 300  # 2 min for admin, 5 min normal
        if current_time - self.last_sync_attempt < min_cooldown:
            remaining = min_cooldown - (current_time - self.last_sync_attempt)
            logger.info(f"⏳ Sync cooldown active: {remaining:.1f}s remaining")
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
    
    async def force_slash_commands_sync(self):
        """PROTECTED Force sync using circuit breaker - SINGLE ATTEMPT ONLY"""
        try:
            if getattr(self, 'emergency_mode', False):
                logger.warning("🚫 Emergency mode active - cannot force sync")
                return False
            
            # Use the same protected sync method
            logger.info("🔄 Admin force sync requested - using protected method...")
            return await self._attempt_resilient_sync()
            
        except Exception as e:
            logger.error(f"❌ Critical force sync error: {e}")
            return False
    
    async def _rebuild_command_tree(self):
        """Rebuild command tree manually to bypass cache issues"""
        try:
            # This will re-register all @bot.tree.command decorated functions
            # Force reload the command definitions
            logger.info("🔄 Rebuilding command definitions...")
            
            # The commands are already defined in the file, just need to ensure they're registered
            commands_count = len(self.tree.get_commands())
            logger.info(f"📝 Found {commands_count} commands in tree")
            
        except Exception as e:
            logger.error(f"Error rebuilding command tree: {e}")
    
    async def _guild_specific_sync(self):
        """Try syncing to specific guilds as fallback"""
        try:
            logger.info("🎯 Attempting guild-specific sync as fallback...")
            synced_guilds = 0
            
            for guild in self.guilds[:3]:  # Try first 3 guilds only
                try:
                    synced = await self.tree.sync(guild=guild)
                    logger.info(f"✅ Synced {len(synced)} commands to {guild.name}")
                    synced_guilds += 1
                    await asyncio.sleep(2)  # Small delay between guild syncs
                except Exception as e:
                    logger.warning(f"⚠️ Failed to sync to {guild.name}: {e}")
            
            if synced_guilds > 0:
                logger.info(f"✅ Guild-specific sync completed for {synced_guilds} guilds")
                self.commands_synced = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Guild-specific sync failed: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Setup error: {e}")
            logger.info("⚠️ Bot will continue with limited functionality")
    
    def can_attempt_sync(self) -> bool:
        """Enhanced check if command sync can be attempted (rate-limit protection)"""
        # Check if sync is completely disabled after multiple failures
        if hasattr(self, 'sync_disabled') and self.sync_disabled:
            logger.warning("🚫 Command sync permanently disabled due to repeated rate limits")
            return False
        
        if self.commands_synced:
            logger.info("✅ Commands already synced successfully, avoiding duplicate sync")
            return False
        
        if self.sync_in_progress:
            logger.info("⚠️ Sync already in progress")
            return False
        
        # Check retry count limit
        if hasattr(self, 'sync_retry_count') and self.sync_retry_count >= 2:
            logger.error("🚫 Maximum sync attempts reached, disabling slash commands")
            self.sync_disabled = True
            return False
        
        current_time = time.time()
        cooldown_time = getattr(self, 'sync_cooldown', 7200)  # Default 2 hours
        last_attempt = getattr(self, 'last_sync_attempt', 0)
        
        if (current_time - last_attempt) < cooldown_time:
            remaining = cooldown_time - (current_time - last_attempt)
            logger.warning(f"⏰ Sync cooldown active: {remaining/3600:.1f} hours remaining")
            return False
        
        return True
    
    async def smart_command_sync(self):
        """REPLACED - Use protected sync method instead"""
        logger.info("🔄 Smart sync redirected to protected method")
        return await self._attempt_resilient_sync()
        logger.info("   • $play_music <song> - Music playback") 
        logger.info("   • $imagen <prompt> - Image generation")
        logger.info("   • $status - System information")
    
    async def connect_lavalink_nodes_safe(self):
        """Enhanced Lavalink connection with intelligent node selection"""
        connected_count = 0
        max_nodes_to_connect = 4  # Increased for better redundancy
        failed_nodes = []
        
        # Sort nodes by priority and add verification
        sorted_nodes = sorted(NODOS_LAVALINK, key=lambda x: x.get('priority', 999))
        
        log_kawaii_info(f"Attempting to connect to {min(max_nodes_to_connect, len(sorted_nodes))} music nodes... *excited*")
        
        for i, node_config in enumerate(sorted_nodes[:max_nodes_to_connect]):
            try:
                log_kawaii_info(f"[{i+1}/{max_nodes_to_connect}] Connecting to {node_config['host']} in {node_config['region']}...")
                
                protocol = "wss" if node_config["secure"] else "ws"
                uri = f"{protocol}://{node_config['host']}:{node_config['port']}"
                
                # Enhanced node configuration
                node = wavelink.Node(
                    uri=uri,
                    password=node_config["password"],
                    identifier=node_config["identifier"],
                    heartbeat=25.0,  # Slightly faster heartbeat
                    retries=1,       # Reduced retries for faster failover
                    resume_timeout=60
                )
                
                # Connect with timeout and proper error handling
                connection_start = time.time()
                await asyncio.wait_for(
                    wavelink.Pool.connect(client=self, nodes=[node]),
                    timeout=8.0  # Reduced timeout for faster failover
                )
                
                connection_time = round((time.time() - connection_start) * 1000)
                self.connected_nodes.append(node_config["identifier"])
                connected_count += 1
                
                log_kawaii_success(f"Connected to {node_config['host']} in {connection_time}ms! *happy dance*")
                
                # Brief pause between connections to avoid overwhelming
                await asyncio.sleep(0.5)
                
                # Break early if we have sufficient nodes
                if connected_count >= 2:
                    log_kawaii_success(f"Sufficient nodes connected ({connected_count}), continuing with startup!")
                    break
                
            except asyncio.TimeoutError:
                failed_nodes.append(f"{node_config['host']} (timeout)")
                logger.warning(f"⏰ Timeout connecting to {node_config['host']}")
                
            except wavelink.InvalidLavalinkVersion as e:
                failed_nodes.append(f"{node_config['host']} (version)")
                logger.warning(f"📦 Version mismatch on {node_config['host']}: {e}")
                
            except wavelink.AuthorizationFailedException as e:
                failed_nodes.append(f"{node_config['host']} (auth)")
                logger.warning(f"🔐 Auth failed on {node_config['host']}: {e}")
                
            except Exception as e:
                failed_nodes.append(f"{node_config['host']} ({type(e).__name__})")
                logger.warning(f"❌ Failed to connect to {node_config['host']}: {e}")
        
        # Final status report
        if connected_count > 0:
            logger.info(f"🎵 Wavelink system operational with {connected_count} nodes")
            if failed_nodes:
                logger.info(f"⚠️ Failed nodes: {', '.join(failed_nodes[:3])}")
        else:
            logger.error("❌ Critical: No Lavalink nodes available - music functionality disabled")
            logger.info("💡 Music commands will show appropriate error messages")
        
        # Store connection stats
        self.wavelink_stats = {
            'connected_nodes': connected_count,
            'failed_nodes': len(failed_nodes),
            'total_attempted': min(max_nodes_to_connect, len(sorted_nodes)),
            'connection_time': datetime.now()
        }
    
    async def on_ready(self):
        """Bot ready event - NO SYNC ATTEMPTS HERE (prevents rate limit loops)"""
        log_kawaii_success(f'{self.user} connected to Discord and ready to be kawaii!')
        log_kawaii_info(f'Active in {len(self.guilds)} guilds with lots of friends!')
        
        # Set bot status immediately
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="🌸 Kawaii commands | Use /help"
            )
        )
        log_kawaii_success("Status configured successfully")
        
        # Setup notification channel for music
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    self.notification_channel = channel
                    break
            if self.notification_channel:
                break
        
        # Display command availability status (slash commands only)
        log_kawaii_info("Only slash commands (/) are active for best experience")
        log_kawaii_info("Use /help for the complete list of kawaii commands")
        log_kawaii_info("Rate limit prevention: Only slash commands available")
        
        # Log resilience status
        if hasattr(self, 'sync_circuit_breaker'):
            can_sync, reason = self.sync_circuit_breaker.can_execute()
            log_kawaii_info(f"Circuit breaker status: {self.sync_circuit_breaker.state.value}")
            if not can_sync:
                log_kawaii_warning(f"Sync blocked: {reason}")
        
        log_kawaii_success("Sakura IA ready for kawaii interactions!")
        
        # Comandos de prefijo eliminados - Solo comandos slash disponibles
        logger.info("🌸 Solo comandos slash (/) disponibles - Comandos de prefijo eliminados completamente")
    
    async def send_long_message(self, channel, content: str, max_length: int = 2000):
        """Send message handling Discord's 2000 character limit by splitting if necessary"""
        if not content:
            return
        
        content = str(content).strip()
        
        # If message is within Discord's limit, send normally
        if len(content) <= max_length:
            try:
                await channel.send(content)
                return
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                return
        
        # Split long message into chunks
        chunks = []
        current_chunk = ""
        
        # Split by lines first to avoid breaking sentences
        lines = content.split('\n')
        
        for line in lines:
            # If adding this line would exceed the limit, save current chunk and start new one
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split by words for chunks that are still too long
                words = chunk.split(' ')
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 > max_length:
                        if current_word_chunk:
                            final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        if current_word_chunk:
                            current_word_chunk += ' ' + word
                        else:
                            current_word_chunk = word
                
                if current_word_chunk:
                    final_chunks.append(current_word_chunk.strip())
        
        # Send all chunks
        for i, chunk in enumerate(final_chunks):
            if chunk.strip():  # Only send non-empty chunks
                try:
                    # Add part indicator for multiple chunks
                    if len(final_chunks) > 1:
                        part_info = f" `({i+1}/{len(final_chunks)})`"
                        # Make sure part info doesn't make chunk too long
                        if len(chunk) + len(part_info) <= max_length:
                            chunk += part_info
                    
                    await channel.send(chunk)
                    
                    # Small delay between chunks to avoid rate limiting
                    if i < len(final_chunks) - 1:
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error sending chunk {i+1}: {e}")
                    # Try without part info if it caused an error
                    try:
                        await channel.send(chunk.replace(part_info, "") if 'part_info' in locals() else chunk)
                    except Exception as e2:
                        logger.error(f"Failed to send chunk even without part info: {e2}")

    async def _retry_sync_with_delay(self, delay_seconds: int = 30):
        """DISABLED - No more retry loops to prevent rate limit cascades"""
        logger.warning("🚫 Retry sync disabled to prevent rate limit loops")
        logger.info("💡 Use $force_sync (admin) or wait for circuit breaker recovery")
        return False
    
    async def on_message(self, message):
        """Enhanced message handling with AI responses, AFK system, and adaptive personality"""
        if message.author.bot:
            return
        
        # Check if channel is ignored for AFK
        if self.afk_manager.is_channel_ignored(message.channel.id):
            return
        
        # Verificar filtros de AutoMod antes de cualquier otro procesamiento
        if message.guild and message.content:
            await self._check_automod_filters(message)
        
        # Process special triggers first (priority system)
        if await self.process_special_triggers(message):
            return
        
        # Save user message for personality detection (only if message has content)
        if message.content and message.guild:
            self.personality_manager.save_user_message(
                message.author.id, 
                message.guild.id, 
                message.content
            )
        
        # Remove AFK status if user sends message
        if self.afk_manager.is_afk(message.author.id):
            afk_data = self.afk_manager.get_afk(message.author.id)
            self.afk_manager.remove_afk(message.author.id)
            
            embed = discord.Embed(
                title="¡Bienvenido de vuelta! 💖",
                description=f"{message.author.mention} ya no está AFK",
                color=COLORES_KAWAII["ROSA_PASTEL"]
            )
            await message.channel.send(embed=embed)
        
        # Check for mentions of AFK users
        for mention in message.mentions:
            if self.afk_manager.is_afk(mention.id):
                afk_data = self.afk_manager.get_afk(mention.id)
                if afk_data:
                    reason = afk_data[2] if afk_data[2] else "sin motivo especificado"
                    await message.channel.send(f"😴 {mention.mention} no está disponible - {reason} owo")
        
        # Handle AI responses for mentions, prefix, and replies
        should_respond = False
        response_trigger = None
        
        # Check if bot is mentioned
        if self.user in message.mentions:
            should_respond = True
            response_trigger = "mention"
        
        # Check for $ prefix only if it's NOT a valid command
        elif message.content.startswith('$'):
            ctx = await self.get_context(message)
            if not ctx.valid:
                should_respond = True
                response_trigger = "prefix"
        
        # Check for (&) prefix - separate from ensemble system
        elif message.content.startswith('&'):
            await self._handle_ampersand_commands(message)
            return
        
        # Check for Sakura prefix commands
        elif message.content.lower().startswith('sakura '):
            command_part = message.content[7:].strip().lower()
            
            # Handle action commands like "sakura hug @user"
            action_commands = ['hug', 'kiss', 'pat', 'poke', 'slap', 'cuddle']
            words = command_part.split()
            
            if words and words[0] in action_commands:
                # Parse action command
                action = words[0]
                target_mention = None
                
                # Find mentioned user
                if message.mentions:
                    target_mention = message.mentions[0]
                
                if target_mention:
                    await self._handle_sakura_action(message, action, target_mention)
                    return
            else:
                # Regular chat with Sakura prefix
                should_respond = True
                response_trigger = "sakura"
        
        # Check if replying to bot's message
        elif (message.reference and 
              message.reference.message_id and 
              hasattr(message.reference, 'resolved') and 
              message.reference.resolved and 
              message.reference.resolved.author == self.user):
            should_respond = True
            response_trigger = "reply"
        
        if should_respond:
            await self._handle_ai_response(message, response_trigger)
            # No procesar comandos tradicionales si ya respondimos con IA
            return
        
        await self.process_commands(message)
    
    async def _check_automod_filters(self, message):
        """Verificar filtros de AutoMod en mensajes"""
        try:
            # Verificar filtros configurados
            result = self.automod_manager.check_message(message.guild.id, message.content)
            
            if result['detected']:
                detected_words = result['words']
                action = result['action']
                
                # Registrar infracción
                self.automod_manager.log_infraction(
                    message.guild.id,
                    message.author.id,
                    message.channel.id,
                    message.content,
                    detected_words,
                    action
                )
                
                # Ejecutar acción según configuración
                if action == 'delete':
                    try:
                        await message.delete()
                        
                        # Crear embed de notificación
                        embed = discord.Embed(
                            title="🚨 Mensaje Filtrado por AutoMod",
                            color=discord.Color.red(),
                            timestamp=datetime.utcnow()
                        )
                        
                        embed.add_field(
                            name="👤 Usuario",
                            value=f"{message.author.mention} ({message.author.display_name})",
                            inline=True
                        )
                        
                        embed.add_field(
                            name="📍 Canal",
                            value=message.channel.mention,
                            inline=True
                        )
                        
                        embed.add_field(
                            name="🔍 Palabras Detectadas",
                            value=f"`{', '.join(detected_words)}`",
                            inline=False
                        )
                        
                        embed.add_field(
                            name="📝 Contenido Original",
                            value=f"```{message.content[:200]}{'...' if len(message.content) > 200 else ''}```",
                            inline=False
                        )
                        
                        embed.set_footer(text=f"Sistema AutoMod • {message.guild.name}")
                        
                        # Enviar notificación
                        notification = await message.channel.send(
                            f"⚠️ {message.author.mention}, tu mensaje fue eliminado por contener palabras prohibidas.",
                            embed=embed,
                            delete_after=10
                        )
                        
                        # Enviar a canal de logs si está configurado
                        config = self.automod_manager.get_config(message.guild.id)
                        if config.get('log_channel_id'):
                            log_channel = self.get_channel(config['log_channel_id'])
                            if log_channel:
                                log_embed = embed.copy()
                                log_embed.title = "📊 Log de AutoMod - Mensaje Eliminado"
                                log_embed.color = discord.Color.orange()
                                await log_channel.send(embed=log_embed)
                        
                        logger.info(f"AutoMod: Mensaje de {message.author} eliminado en {message.guild.name} - Palabras: {detected_words}")
                        
                    except discord.Forbidden:
                        logger.warning(f"Sin permisos para eliminar mensaje en {message.channel.name}")
                    except discord.NotFound:
                        pass  # Mensaje ya eliminado
                    
                elif action == 'warn':
                    # Solo advertir sin eliminar
                    embed = discord.Embed(
                        title="⚠️ Advertencia de AutoMod",
                        description=f"{message.author.mention}, tu mensaje contiene palabras no permitidas: `{', '.join(detected_words)}`",
                        color=discord.Color.orange()
                    )
                    
                    await message.channel.send(embed=embed, delete_after=5)
                    logger.info(f"AutoMod: Advertencia enviada a {message.author} en {message.guild.name}")
                    
        except Exception as e:
            logger.error(f"Error en verificación de AutoMod: {e}")
    
    async def on_auto_moderation_action_execution(self, payload):
        """Evento que se ejecuta cuando AutoMod nativo toma una acción"""
        try:
            # Obtener información del evento
            guild = payload.guild
            user = payload.member
            channel = payload.channel
            rule_id = payload.rule_id
            content = getattr(payload, 'content', 'Contenido no disponible')
            
            # Intentar obtener la regla que se activó
            try:
                rule = await guild.fetch_automod_rule(rule_id)
                rule_name = rule.name
            except:
                rule_name = f"Regla ID: {rule_id}"
            
            # Crear embed informativo
            embed = discord.Embed(
                title="🛡️ AutoMod Nativo - Acción Ejecutada",
                color=discord.Color.red(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(
                name="👤 Usuario",
                value=f"{user.mention} ({user.display_name})",
                inline=True
            )
            
            embed.add_field(
                name="📋 Regla Activada",
                value=rule_name,
                inline=True
            )
            
            embed.add_field(
                name="📍 Canal",
                value=channel.mention if channel else "Canal no disponible",
                inline=True
            )
            
            embed.add_field(
                name="📝 Contenido Bloqueado",
                value=f"```{content[:500]}{'...' if len(content) > 500 else ''}```",
                inline=False
            )
            
            embed.set_footer(text=f"AutoMod Nativo • {guild.name}")
            
            # Enviar mensaje al canal donde ocurrió la infracción
            if channel and hasattr(channel, 'send'):
                try:
                    await channel.send(
                        f"🛡️ {user.mention}, tu mensaje fue procesado por AutoMod nativo.",
                        embed=embed,
                        delete_after=15
                    )
                    logger.info(f"Notificación de AutoMod nativo enviada en {channel.name}")
                except discord.Forbidden:
                    logger.warning(f"Sin permisos para enviar mensaje en {channel.name}")
                except Exception as e:
                    logger.error(f"Error al enviar notificación de AutoMod nativo: {e}")
            
            # Enviar a canal de logs si está configurado
            config = self.automod_manager.get_config(guild.id)
            if config.get('log_channel_id'):
                log_channel = self.get_channel(config['log_channel_id'])
                if log_channel:
                    log_embed = embed.copy()
                    log_embed.title = "📊 Log AutoMod Nativo"
                    log_embed.color = discord.Color.blue()
                    await log_channel.send(embed=log_embed)
            
            # Log detallado del evento
            logger.info(
                f"AutoMod nativo activado: Usuario={user.display_name}, "
                f"Canal={channel.name if channel else 'N/A'}, "
                f"Regla={rule_name}, Contenido={content[:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Error en evento de AutoMod nativo: {e}")
    
    async def _handle_sakura_action(self, message, action, target_user):
        """Handle Sakura action commands like 'Sakura hug @user'"""
        try:
            gif_url = await self.gif_provider.get_gif(action)
            
            action_messages = {
                'hug': {
                    'title': '🤗 ¡Abrazo kawaii enviado por Sakura!',
                    'description': f'¡Sakura abraza cariñosamente a {target_user.mention} por petición de {message.author.mention}! *hug hug* UwU 💖✨',
                    'button_text': 'Devolver abrazo'
                },
                'kiss': {
                    'title': '😘 ¡Besito kawaii enviado por Sakura!',
                    'description': f'¡Sakura le manda un besito volador a {target_user.mention} por petición de {message.author.mention}! Muah~ 💖✨',
                    'button_text': 'Devolver besito'
                },
                'pat': {
                    'title': '✋ ¡Palmaditas kawaii de Sakura!',
                    'description': f'¡Sakura le da palmaditas cariñosas a {target_user.mention} por petición de {message.author.mention}! *pat pat* UwU 💖',
                    'button_text': 'Devolver caricias'
                },
                'poke': {
                    'title': '👆 ¡Cosquillitas kawaii de Sakura!',
                    'description': f'¡Sakura le hace cosquillitas a {target_user.mention} por petición de {message.author.mention}! *poke poke* >w< ✨',
                    'button_text': 'Devolver cosquillitas'
                },
                'cuddle': {
                    'title': '🥰 ¡Mimos kawaii de Sakura!',
                    'description': f'¡Sakura le da mimos tiernos a {target_user.mention} por petición de {message.author.mention}! *cuddle cuddle* UwU 💖',
                    'button_text': 'Devolver mimos'
                }
            }
            
            action_data = action_messages.get(action, action_messages['hug'])
            
            embed = discord.Embed(
                title=action_data['title'],
                description=action_data['description'],
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if gif_url:
                embed.set_image(url=gif_url)
            
            # Create response button
            class ActionView(discord.ui.View):
                def __init__(self):
                    super().__init__(timeout=60)
                    
                @discord.ui.button(emoji="❤️", style=discord.ButtonStyle.success, label=action_data['button_text'])
                async def return_action(self, interaction: discord.Interaction, button: discord.ui.Button):
                    if interaction.user.id != target_user.id:
                        await interaction.response.send_message("¡Solo la persona mencionada puede devolver la acción! UwU", ephemeral=True)
                        return
                    
                    return_messages = {
                        'hug': f"¡{target_user.mention} devuelve el abrazo cariñosamente! *hug hug back* UwU ✨",
                        'kiss': f"¡{target_user.mention} devuelve el besito! Muah muah~ UwU ✨",
                        'pat': f"¡{target_user.mention} devuelve las palmaditas! *pat pat back* UwU ✨",
                        'poke': f"¡{target_user.mention} devuelve las cosquillitas! *poke poke back* >w< ✨",
                        'cuddle': f"¡{target_user.mention} devuelve los mimos! *cuddle cuddle back* UwU ✨"
                    }
                    
                    embed = discord.Embed(
                        title="🌸 ¡Acción devuelta! 💕",
                        description=return_messages.get(action, return_messages['hug']),
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    await interaction.response.edit_message(embed=embed, view=None)
            
            view = ActionView()
            await message.channel.send(embed=embed, view=view)
            
        except Exception as e:
            logger.error(f"Error in Sakura action: {e}")
            await message.channel.send("¡Upsi~ algo pasó con mi cerebrito, pero ya vuelvo UwU! 💔")

    def _check_user_cooldown(self, user_id: int) -> tuple[bool, float]:
        """Check if user is on cooldown for AI interactions"""
        current_time = time.time()
        
        if user_id in self.user_cooldowns:
            time_since_last = current_time - self.user_cooldowns[user_id]
            if time_since_last < self.ai_cooldown_duration:
                remaining = self.ai_cooldown_duration - time_since_last
                return False, remaining
        
        return True, 0.0
    
    def _update_user_cooldown(self, user_id: int):
        """Update user's last interaction time"""
        self.user_cooldowns[user_id] = time.time()
    


    async def _handle_ai_response(self, message, trigger_type):
        """Handle AI response with adaptive personality using the exact format"""
        try:
            # Check user cooldown first
            can_respond, remaining_time = self._check_user_cooldown(message.author.id)
            if not can_respond:
                # Send cooldown message with Sakura personality
                embed = discord.Embed(
                    title="🌸 ¡Un momentito, mi amor! >.<",
                    description=f"*susurra tímidamente* Dame {remaining_time:.0f} segunditos más para recuperarme~ uwu\n\n"
                               f"💭 Mientras tanto puedes usar otros comandos o hablar con otros users owo ✨",
                    color=0xFFB6C1  # Rosa pastel
                )
                embed.set_footer(text="💖 Sakura IA necesita un descansito entre conversaciones owo")
                await message.reply(embed=embed)
                return
            
            # Update user cooldown
            self._update_user_cooldown(message.author.id)
            # Get user personality safely
            if message.guild:
                personality = self.personality_manager.get_user_personality(
                    message.author.id
                )
            else:
                personality = 'normal'
            
            # Prepare message content
            content = message.content
            if trigger_type == "mention":
                content = content.replace(f'<@{self.user.id}>', '').strip()
            elif trigger_type == "prefix":
                content = content[1:].strip()
            elif trigger_type == "sakura":
                content = content[7:].strip()
            
            if not content:
                content = "hola"
            
            # Get AI response using enhanced system
            logger.info(f"Getting AI response for user {message.author.id}: {content[:100]}")
            ai_response = await bot.ai_provider.get_premium_ai_response(content, message.author.id, None, "auto")
                
            if not ai_response:
                logger.error("All AI providers failed, using fallback response")
                ai_response = "¡Hola! Soy Sakura IA y estoy aquí para ayudarte. ¿En qué puedo asistirte hoy?"
            else:
                logger.info("✅ AI response generated successfully")
            
            # Adapt the response based on personality
            personality_adapted_response = bot.ai_provider._adapt_ai_response_to_personality(
                ai_response, personality, message.author.display_name
            )
            
            # Truncate AI response if too long (Discord limit handling)
            max_ai_content = 1800  # Leave room for formatting
            if len(personality_adapted_response) > max_ai_content:
                personality_adapted_response = personality_adapted_response[:max_ai_content] + "...\n\n*[Respuesta truncada por límite de caracteres uwu~]*"
            
            # Create the formatted response with Sakura IA branding
            formatted_response = f"""🌸 **Sakura IA responde**

{personality_adapted_response}

> **Pregunta de {message.author.display_name}**: {content[:100]}{'...' if len(content) > 100 else ''}"""
            
            # Ensure final response doesn't exceed Discord limit
            if len(formatted_response) > 1990:
                # Emergency truncation
                truncated_content = personality_adapted_response[:1500] + "...\n\n*[Respuesta truncada uwu~]*"
                formatted_response = f"""🌸 **Sakura IA responde**

{truncated_content}

> **Pregunta de {message.author.display_name}**: {content[:50]}..."""
            
            # Save conversation
            self.personality_manager.save_conversation(
                message.author.id, message.guild.id, content, personality_adapted_response
            )
            
            # Save ensemble conversation to PostgreSQL
            if hasattr(self, 'postgresql_manager'):
                response_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
                await self.postgresql_manager.save_ensemble_conversation(
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else 0,
                    message=content,
                    response=personality_adapted_response,
                    ai_provider="ensemble",
                    response_time_ms=response_time_ms
                )
            
            # Send response with proper length handling
            await self.send_long_message(message.channel, formatted_response)
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            try:
                # Try to get personality safely
                if message.guild:
                    personality = self.personality_manager.get_user_personality(
                        message.author.id
                    )
                else:
                    personality = 'normal'
            except:
                personality = 'normal'
                
            error_responses = {
                'femboy': "¡Hola preciosa! UwU Mi cerebrito de Sakura IA está un poquito cansado ahora, pero siempre estoy aquí para ti~ 💖 ¿Quieres intentar de nuevo más tarde? >w<",
                'normal': "¡Hola! Soy Sakura IA y estoy aquí para ayudarte. Hubo un pequeño problema, pero puedes intentar de nuevo o usar otros comandos."
            }
            error_msg = error_responses.get(personality, error_responses['normal'])
            await message.channel.send(f"🌸 **Sakura IA responde**\n\n{error_msg}")
    
    async def _handle_ampersand_commands(self, message):
        """Handle (&) prefix commands separate from ensemble system"""
        command_content = message.content[1:].strip()  # Remove & prefix
        parts = command_content.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if command == "search" or command == "buscar":
                if not args:
                    embed = discord.Embed(
                        title="🔍 Búsqueda Kawaii",
                        description="**Uso:** `&search [tu consulta]`\n\n"
                                   "**Ejemplo:** `&search gatos kawaii`",
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    await message.reply(embed=embed)
                    return
                
                await self._handle_ampersand_search(message, args)
                
            elif command == "img" or command == "imagen":
                if not args:
                    embed = discord.Embed(
                        title="🖼️ Búsqueda de Imágenes",
                        description="**Uso:** `&img [consulta]`\n\n"
                                   "**Ejemplo:** `&img anime kawaii`",
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    await message.reply(embed=embed)
                    return
                
                await self._handle_ampersand_image_search(message, args)
                
            elif command == "yt" or command == "youtube":
                if not args:
                    embed = discord.Embed(
                        title="🎵 Búsqueda de YouTube",
                        description="**Uso:** `&yt [consulta]`\n\n"
                                   "**Ejemplo:** `&yt música kawaii`",
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    await message.reply(embed=embed)
                    return
                
                await self._handle_ampersand_youtube_search(message, args)
                
            elif command == "stats" or command == "estadisticas":
                await self._handle_ampersand_stats(message)
                
            elif command == "help" or command == "ayuda":
                await self._handle_ampersand_help(message)
                
            else:
                embed = discord.Embed(
                    title="❓ Comando No Encontrado",
                    description=f"El comando `&{command}` no existe.\n\n"
                               "Usa `&help` para ver comandos disponibles.",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
                await message.reply(embed=embed)
                
        except Exception as e:
            logger.error(f"Error in ampersand command '{command}': {e}")
            embed = discord.Embed(
                title="💔 Error",
                description="Hubo un error procesando tu comando.",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await message.reply(embed=embed)
    
    async def _handle_ampersand_search(self, message, query):
        """Handle &search command with PostgreSQL logging"""
        embed = discord.Embed(
            title="🔍 Buscando...",
            description=f"Realizando búsqueda web para: **{query}**",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_msg = await message.reply(embed=embed)
        
        try:
            search_results = []
            for result in google_search(query, num_results=10):
                search_results.append(result)
            
            # Save to PostgreSQL
            if hasattr(self, 'postgresql_manager'):
                self.postgresql_manager.save_search(
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else 0,
                    search_type="web_search",
                    query=query,
                    results_count=len(search_results)
                )
            
            results_embed = discord.Embed(
                title="🔍 Resultados de Búsqueda Web",
                description=f"**Consulta:** {query}\n**Resultados:** {len(search_results)}",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            for i, result in enumerate(search_results[:5]):
                results_embed.add_field(
                    name=f"#{i+1}",
                    value=f"[{result[:50]}...]({result})" if len(result) > 50 else f"[Resultado]({result})",
                    inline=False
                )
            
            await processing_msg.edit(embed=results_embed)
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            error_embed = discord.Embed(
                title="💔 Error en Búsqueda",
                description="No se pudo completar la búsqueda web.",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await processing_msg.edit(embed=error_embed)
    
    async def _handle_ampersand_image_search(self, message, query):
        """Handle &img command with PostgreSQL logging"""
        embed = discord.Embed(
            title="🖼️ Buscando imágenes...",
            description=f"Búsqueda de imágenes: **{query}**",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_msg = await message.reply(embed=embed)
        
        try:
            results = await self.search_manager.search_images(query, max_results=20)
            
            # Save to PostgreSQL
            if hasattr(self, 'postgresql_manager'):
                self.postgresql_manager.save_search(
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else 0,
                    search_type="image_search",
                    query=query,
                    results_count=len(results)
                )
            
            if results:
                await self.search_manager.send_search_results(
                    message.channel, query, results, "image", processing_msg
                )
            else:
                error_embed = discord.Embed(
                    title="💔 Sin Resultados",
                    description=f"No se encontraron imágenes para: **{query}**",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
                await processing_msg.edit(embed=error_embed)
                
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            error_embed = discord.Embed(
                title="💔 Error en Búsqueda",
                description="No se pudo completar la búsqueda de imágenes.",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await processing_msg.edit(embed=error_embed)
    
    async def _handle_ampersand_youtube_search(self, message, query):
        """Handle &yt command with PostgreSQL logging"""
        embed = discord.Embed(
            title="🎵 Buscando en YouTube...",
            description=f"Búsqueda de videos: **{query}**",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_msg = await message.reply(embed=embed)
        
        try:
            results = await self.search_manager.search_youtube(query, max_results=20)
            
            # Save to PostgreSQL
            if hasattr(self, 'postgresql_manager'):
                self.postgresql_manager.save_search(
                    user_id=message.author.id,
                    guild_id=message.guild.id if message.guild else 0,
                    search_type="youtube_search",
                    query=query,
                    results_count=len(results)
                )
            
            if results:
                await self.search_manager.send_search_results(
                    message.channel, query, results, "youtube", processing_msg
                )
            else:
                error_embed = discord.Embed(
                    title="💔 Sin Resultados",
                    description=f"No se encontraron videos para: **{query}**",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
                await processing_msg.edit(embed=error_embed)
                
        except Exception as e:
            logger.error(f"Error in YouTube search: {e}")
            error_embed = discord.Embed(
                title="💔 Error en Búsqueda",
                description="No se pudo completar la búsqueda de YouTube.",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await processing_msg.edit(embed=error_embed)
    
    async def _handle_ampersand_stats(self, message):
        """Handle &stats command to show user search statistics"""
        try:
            if hasattr(self, 'postgresql_manager'):
                stats = self.postgresql_manager.get_user_search_stats(
                    message.author.id,
                    message.guild.id if message.guild else 0
                )
                
                embed = discord.Embed(
                    title="📊 Tus Estadísticas de Búsqueda",
                    description=f"Estadísticas para {message.author.mention}",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
                
                if stats:
                    total_searches = sum(stats.values())
                    embed.add_field(
                        name="🔍 Total de Búsquedas",
                        value=str(total_searches),
                        inline=True
                    )
                    
                    for search_type, count in stats.items():
                        type_name = {
                            "web_search": "🌐 Web",
                            "image_search": "🖼️ Imágenes", 
                            "youtube_search": "🎵 YouTube"
                        }.get(search_type, search_type)
                        
                        embed.add_field(
                            name=type_name,
                            value=str(count),
                            inline=True
                        )
                else:
                    embed.add_field(
                        name="📝 Sin Datos",
                        value="No has realizado búsquedas aún.",
                        inline=False
                    )
                
                await message.reply(embed=embed)
            else:
                embed = discord.Embed(
                    title="💔 Error",
                    description="Sistema de estadísticas no disponible.",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
                await message.reply(embed=embed)
                
        except Exception as e:
            logger.error(f"Error in stats command: {e}")
            embed = discord.Embed(
                title="💔 Error",
                description="No se pudieron obtener las estadísticas.",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await message.reply(embed=embed)
    
    async def _handle_ampersand_help(self, message):
        """Handle &help command"""
        embed = discord.Embed(
            title="🌸 Comandos (&) Disponibles",
            description="Sistema de comandos separado del chat IA",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        embed.add_field(
            name="🔍 &search [consulta]",
            value="Búsqueda web general",
            inline=False
        )
        
        embed.add_field(
            name="🖼️ &img [consulta]",
            value="Búsqueda de imágenes",
            inline=False
        )
        
        embed.add_field(
            name="🎵 &yt [consulta]",
            value="Búsqueda en YouTube",
            inline=False
        )
        
        embed.add_field(
            name="📊 &stats",
            value="Ver tus estadísticas de búsqueda",
            inline=False
        )
        
        embed.add_field(
            name="❓ &help",
            value="Mostrar esta ayuda",
            inline=False
        )
        
        embed.set_footer(text="Estos comandos son independientes del chat IA y guardan datos en PostgreSQL")
        
        await message.reply(embed=embed)
    
    def _adapt_ai_response_to_personality(self, ai_response: str, personality: str, username: str) -> str:
        """Adapt AI response to enhance kawaii personality consistently"""
        if not ai_response:
            return "¡Yaaay, hola hola~! 🌸✨ Soy Sakura IA, tu amiguita virtual más adorable UwU~ ¿En qué puedo ayudarte hoy, senpai~? >w< ¡Estoy súper emocionada de charlar contigo y darte toda mi sakura-sabiduría~! 💖"
        
        # Enhanced kawaii personality for all responses
        if personality == 'femboy' or personality == 'kawaii':
            # Add ultra kawaii expressive elements
            if not any(x in ai_response.lower() for x in ['uwu', 'owo', '~', 'nyan', 'senpai']):
                ai_response = f"¡Hola hola~ {username}-senpai! " + ai_response
            if not any(x in ai_response for x in ['💖', '🌸', '✨', '>w<', 'UwU']):
                ai_response += " UwU 🌸💖"
            # Add more kawaii expressions
            ai_response = ai_response.replace("!", "! >w<").replace("?", "? nyan~")
                
        # For 'normal' personality, still keep kawaii but tone it down slightly
        else:
            # Keep kawaii but less intense
            if not any(greeting in ai_response.lower() for greeting in ["hola", "soy sakura"]):
                ai_response = f"¡Hola {username}! Soy Sakura IA~ " + ai_response
            if not any(x in ai_response for x in ['🌸', '💖', '✨']):
                ai_response += " 🌸"
            
        return ai_response

# Initialize bot
bot = NekotinaBot()

# Decorator para convertir slash commands en prefix commands automáticamente
def create_prefix_command(slash_func, command_name=None):
    """Decorator que convierte una función de slash command en prefix command"""
    if command_name is None:
        command_name = slash_func.__name__.replace('_command', '').replace('_', '')
    
    # Crear función wrapper que simula la interfaz de slash command
    async def prefix_wrapper(ctx, *args):
        # Crear mock interaction que funciona como slash command
        mock_interaction = type('MockInteraction', (), {})()
        mock_interaction.user = ctx.author
        mock_interaction.guild = ctx.guild
        mock_interaction.channel = ctx.channel
        mock_interaction.response = type('MockResponse', (), {})()
        mock_interaction.followup = type('MockFollowup', (), {})()
        
        # Variables para capturar respuestas
        response_sent = False
        response_data = {}
        
        # Mock response methods
        async def mock_send_message(content=None, embed=None, ephemeral=False, view=None, file=None):
            nonlocal response_sent, response_data
            response_sent = True
            response_data = {'content': content, 'embed': embed, 'view': view, 'file': file}
            
            if embed:
                await ctx.send(embed=embed, view=view, file=file)
            else:
                await ctx.send(content, view=view, file=file)
        
        async def mock_defer():
            nonlocal response_sent
            response_sent = True
            # En prefix commands, no necesitamos defer
            pass
        
        async def mock_edit_message(content=None, embed=None, view=None):
            if embed:
                await ctx.send(embed=embed, view=view)
            else:
                await ctx.send(content, view=view)
        
        # Configurar mocks
        mock_interaction.response.send_message = mock_send_message
        mock_interaction.response.defer = mock_defer
        mock_interaction.response.edit_message = mock_edit_message
        mock_interaction.response.is_done = lambda: response_sent
        mock_interaction.followup.send = mock_send_message
        mock_interaction.followup.edit_message = mock_edit_message
        
        try:
            # Intentar ejecutar con parámetros como slash command
            import inspect
            sig = inspect.signature(slash_func)
            params = list(sig.parameters.keys())[1:]  # Omitir 'interaction'
            
            # Convertir argumentos de texto a tipos apropiados
            converted_args = []
            for i, param_name in enumerate(params):
                if i < len(args):
                    arg_value = args[i]
                    param = sig.parameters[param_name]
                    
                    # Conversión básica de tipos
                    if param.annotation == int:
                        try:
                            arg_value = int(arg_value)
                        except:
                            arg_value = 0
                    elif param.annotation == float:
                        try:
                            arg_value = float(arg_value)
                        except:
                            arg_value = 0.0
                    elif param.annotation == bool:
                        arg_value = arg_value.lower() in ('true', '1', 'yes', 'sí', 'si')
                    # Para menciones de usuario
                    elif 'discord.Member' in str(param.annotation) or 'discord.User' in str(param.annotation):
                        if ctx.message.mentions:
                            arg_value = ctx.message.mentions[0]
                        else:
                            # Intentar encontrar usuario por nombre/ID
                            try:
                                if arg_value.isdigit():
                                    arg_value = await bot.fetch_user(int(arg_value))
                                else:
                                    # Buscar por nombre en el servidor
                                    member = discord.utils.get(ctx.guild.members, name=arg_value)
                                    if member:
                                        arg_value = member
                                    else:
                                        arg_value = ctx.author  # Fallback al usuario que escribió
                            except:
                                arg_value = ctx.author
                    
                    converted_args.append(arg_value)
                else:
                    # Parámetro opcional o valor por defecto
                    if param.default != inspect.Parameter.empty:
                        converted_args.append(param.default)
                    else:
                        # Valores por defecto según tipo
                        if param.annotation == int:
                            converted_args.append(0)
                        elif param.annotation == float:
                            converted_args.append(0.0)
                        elif param.annotation == bool:
                            converted_args.append(False)
                        elif param.annotation == str:
                            converted_args.append("")
                        else:
                            converted_args.append(None)
            
            # Ejecutar función slash original
            await slash_func(mock_interaction, *converted_args)
            
        except Exception as e:
            logger.error(f"Error en prefix command {command_name}: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=f"Hubo un error ejecutando el comando: {str(e)[:200]}",
                color=0xFF0000
            )
            await ctx.send(embed=embed)
    
    # Configurar el comando prefix - verificar si ya existe
    prefix_wrapper.__name__ = f"{command_name}_prefix"
    
    # Verificar si el comando ya existe antes de registrarlo
    if command_name in [cmd.name for cmd in bot.commands]:
        logger.warning(f"Command {command_name} already exists as prefix command, skipping")
        return None
    
    # Verificar si existe como alias
    for cmd in bot.commands:
        if command_name in getattr(cmd, 'aliases', []):
            logger.warning(f"Command {command_name} already exists as alias, skipping")
            return None
    
    return bot.command(name=command_name, aliases=[command_name.lower()])(prefix_wrapper)



# Comandos de prefijo completamente removidos - Solo comandos slash activos

# Roleplay Commands
@bot.tree.command(name="act", description="Realizar una acción sin objetivo")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(accion="La acción que quieres realizar")
async def act_command(interaction: discord.Interaction, accion: str):
    """Act command for solo actions"""
    try:
        accion = accion.lower()
        
        if accion not in bot.gif_provider.act_actions:
            available_actions = ", ".join(bot.gif_provider.act_actions)
            embed = discord.Embed(
                title="🥺 *se disculpa tímidamente* Lo siento...",
                description=f"*murmura* No conozco esa acción... ¿podrías elegir entre estas? {available_actions}\n\n*se esconde* Perdón por no entender... 🌸",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        await interaction.response.defer()
        
        gif_url = await bot.gif_provider.get_gif(accion)
        description = bot.gif_provider.action_descriptions.get(accion, accion)
        
        if gif_url:
            embed = discord.Embed(
                description=f"**{interaction.user.mention} {description}**",
                color=COLORES_KAWAII["ROSA_PASTEL"]
            )
            embed.set_image(url=gif_url)
        else:
            embed = discord.Embed(
                title="🥺 *se sonroja de vergüenza*",
                description=f'*susurra* Ay... no pude encontrar una imagen para "{accion}"... \n\nPerdón... tal vez puedas intentar con otra acción... 💫',
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
        
        await interaction.followup.send(embed=embed)
    except Exception as e:
        logger.error(f"Error in act command: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message("*se disculpa nerviosamente* 🥺 Ay no... algo salió mal... ¿podrías intentar de nuevo? Lo siento mucho...", ephemeral=True)
        else:
            await interaction.followup.send("*susurra tristemente* 🥺 Perdón... hubo un problemita... ¿podrías intentar otra vez? Lo siento tanto...", ephemeral=True)

@bot.tree.command(name="interact", description="Interactuar con otro usuario")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    accion="La acción que quieres realizar",
    usuario="El usuario con quien interactuar"
)
async def interact_command(interaction: discord.Interaction, accion: str, usuario: discord.User):
    """Interact command for actions with target"""
    accion = accion.lower()
    
    if accion not in bot.gif_provider.interact_actions:
        available_actions = ", ".join(bot.gif_provider.interact_actions)
        embed = discord.Embed(
            title="❌ Acción no válida",
            description=f"Acciones de interacción disponibles: {available_actions}",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="❌ Error",
            description="No puedes interactuar contigo mismo. Usa `/act` para acciones individuales.",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    await interaction.response.defer()
    
    gif_url = await bot.gif_provider.get_gif(accion)
    description = bot.gif_provider.action_descriptions.get(accion, accion)
    
    if gif_url:
        embed = discord.Embed(
            description=f"**{interaction.user.mention} {description} a {usuario.mention}**",
            color=COLORES_KAWAII["ROSA_PASTEL"]
        )
        embed.set_image(url=gif_url)
    else:
        embed = discord.Embed(
            title="❌ Error",
            description=f'Lo siento, no encontré un GIF para "{accion}".',
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
    
    await interaction.followup.send(embed=embed)

# Fun Commands
@bot.tree.command(name="catfact", description="Obtener un dato curioso sobre gatos")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def catfact_command(interaction: discord.Interaction):
    """Get a cat fact"""
    fact = random.choice(bot.cat_facts)
    embed = discord.Embed(
        title="🐱 *susurra* Un dato sobre gatitos... si te interesa 🌸",
        description=f"Eh... bueno... {fact} \n\n*se esconde tímidamente* Espero que te guste... 💫",
        color=COLORES_KAWAII["ALERTA_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="dogfact", description="Obtener un dato curioso sobre perros")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def dogfact_command(interaction: discord.Interaction):
    """Get a dog fact"""
    fact = random.choice(bot.dog_facts)
    embed = discord.Embed(
        title="🐶 *murmura suavemente* Algo sobre perritos... 🌸",
        description=f"Um... te comparto esto... {fact} \n\n*mira hacia otro lado* Espero no molestarte... ✨",
        color=COLORES_KAWAII["ROSA_DULCE"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="lucky", description="Probar tu suerte del día")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def lucky_command(interaction: discord.Interaction):
    """Get daily luck"""
    luck_percentage = random.randint(1, 100)
    
    if luck_percentage >= 90:
        message = "*susurra* Oh... parece que tienes muchísima suerte hoy... ✨"
        color = 0xFFD700
    elif luck_percentage >= 70:
        message = "*sonríe tímidamente* Tienes bastante suerte... 🍀"
        color = 0x32CD32
    elif luck_percentage >= 50:
        message = "*habla suavemente* Tu suerte está... está bien 😊"
        color = 0x87CEEB
    elif luck_percentage >= 30:
        message = "*preocupada* Perdón... no parece un día muy afortunado... 😅"
        color = 0xFFA500
    else:
        message = "*susurra preocupada* Por favor... ten mucho cuidado hoy... 😬"
        color = 0xFF6B6B
    
    embed = discord.Embed(
        title="🎲 *timidamente* Tu suerte... si quieres saberla...",
        description=f"**{luck_percentage}%** - {message}",
        color=color
    )
    await interaction.response.send_message(embed=embed)



# Game Commands
@bot.tree.command(name="roll", description="Lanzar dados")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(dados="Formato: XdY (ej: 2d6 para 2 dados de 6 caras)")
async def roll_command(interaction: discord.Interaction, dados: str = "1d6"):
    """Roll dice"""
    try:
        if 'd' not in dados:
            raise ValueError("Formato inválido")
        
        num_dice, sides = map(int, dados.split('d'))
        
        if num_dice > 10 or sides > 100:
            embed = discord.Embed(
                title="🥺 *se disculpa* Lo siento...",
                description="*susurra nerviosa* Perdón... solo puedo lanzar hasta 10 dados de 100 caras... ¿está bien así? 💫",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        results = [random.randint(1, sides) for _ in range(num_dice)]
        total = sum(results)
        
        embed = discord.Embed(
            title="🎲 *susurra* Aquí están tus dados... 🌸",
            description=(
                f"╔═══════════════════════╗\n"
                f"  🎯 **Lanzamiento: {dados}**\n"
                f"╚═══════════════════════╝"
            ),
            color=COLORES_KAWAII["MORADO_KAWAII"]
        )
        
        # Visual dice representation
        dice_visuals = {
            1: "⚀", 2: "⚁", 3: "⚂", 
            4: "⚃", 5: "⚄", 6: "⚅"
        }
        
        visual_results = []
        for result in results:
            if result <= 6:
                visual_results.append(dice_visuals.get(result, str(result)))
            else:
                visual_results.append(f"`{result}`")
        
        embed.add_field(
            name="🎲 *murmura* Los resultados...",
            value=" ".join(visual_results),
            inline=False
        )
        embed.add_field(
            name="📊 Resultados Detallados",
            value=f"```{', '.join(map(str, results))}```",
            inline=True
        )
        embed.add_field(
            name="✨ Total",
            value=f"```🎯 {total}```",
            inline=True
        )
        await interaction.response.send_message(embed=embed)
        
    except ValueError:
        embed = discord.Embed(
            title="❌ Error",
            description="Formato inválido. Usa XdY (ej: 2d6)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)



@bot.tree.command(name="hangman", description="Jugar al ahorcado")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def hangman_command(interaction: discord.Interaction):
    """Start hangman game"""
    game = bot.game_manager.create_hangman_game(interaction.user.id)
    
    embed = discord.Embed(
        title="🎯 Juego del Ahorcado",
        description=f"**Palabra:** {' '.join(game['display'])}\n**Intentos incorrectos:** {game['wrong_guesses']}/{game['max_wrong']}\n**Letras usadas:** Ninguna",
        color=COLORES_KAWAII["MORADO_KAWAII"]
    )
    embed.set_footer(text="Usa 'S guess <letra>' para adivinar una letra")
    await interaction.response.send_message(embed=embed)

@bot.command(name="guess")
async def guess_command(ctx, letter: str = None):
    """Guess letter in hangman"""
    if letter is None or len(letter) != 1 or not letter.isalpha():
        embed = discord.Embed(
            title="❌ Error",
            description="Proporciona una sola letra",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await ctx.send(embed=embed)
        return
    
    result = bot.game_manager.guess_letter(ctx.author.id, letter)
    
    if result is None:
        embed = discord.Embed(
            title="❌ Error",
            description="No tienes un juego de ahorcado activo. Usa `/hangman` para empezar.",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await ctx.send(embed=embed)
        return
    
    game = result['game']
    status = result['status']
    
    if status == 'already_guessed':
        embed = discord.Embed(
            title="⚠️ Letra ya usada",
            description=f"Ya has usado la letra '{letter.upper()}'",
            color=COLORES_KAWAII["ALERTA_KAWAII"]
        )
    elif status == 'won':
        embed = discord.Embed(
            title="🎉 ¡Ganaste!",
            description=f"¡Felicitaciones! La palabra era: **{game['word']}**",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
    elif status == 'lost':
        embed = discord.Embed(
            title="💀 ¡Perdiste!",
            description=f"Se acabaron los intentos. La palabra era: **{game['word']}**",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
    else:
        status_emoji = "✅" if status == 'correct' else "❌"
        embed = discord.Embed(
            title=f"{status_emoji} {'¡Correcto!' if status == 'correct' else '¡Incorrecto!'}",
            description=f"**Palabra:** {' '.join(game['display'])}\n**Intentos incorrectos:** {game['wrong_guesses']}/{game['max_wrong']}\n**Letras usadas:** {', '.join(sorted(game['guessed']))}",
            color=COLORES_KAWAII["EXITO_KAWAII"] if status == 'correct' else 0xFF6B6B
        )
    
    await ctx.send(embed=embed)

# Image Edit Commands
@bot.tree.command(name="blur", description="Aplicar desenfoque a una imagen")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(imagen="URL de la imagen o adjunta una imagen")
async def blur_command(interaction: discord.Interaction, imagen: str = None):
    """Blur an image"""
    image_url = imagen
    
    # Check for attachments if no URL provided
    if image_url is None and interaction.message and interaction.message.attachments:
        image_url = interaction.message.attachments[0].url
    elif image_url is None:
        embed = discord.Embed(
            title="❌ Error",
            description="Proporciona una URL de imagen o adjunta una imagen",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    await interaction.response.defer()
    
    result = await ImageEditor.apply_filter(image_url, 'blur')
    if result:
        file = discord.File(result, filename="blurred_image.png")
        embed = discord.Embed(
            title="🌫️ Imagen Desenfocada",
            color=COLORES_KAWAII["CELESTE_KAWAII"]
        )
        embed.set_image(url="attachment://blurred_image.png")
        await interaction.followup.send(embed=embed, file=file)
    else:
        embed = discord.Embed(
            title="❌ Error",
            description="No se pudo procesar la imagen",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)

# AFK System Commands
@bot.tree.command(name="afk", description="Establecer estado AFK")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(motivo="Motivo de tu ausencia")
async def afk_command(interaction: discord.Interaction, motivo: str = None):
    """Set AFK status"""
    if motivo is None:
        motivo = "No hay razón especificada"
    
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    bot.afk_manager.set_afk(interaction.user.id, guild_id, motivo)
    
    embed = discord.Embed(
        title="😴 Estado AFK Establecido",
        description=f"{interaction.user.mention} ahora está AFK\n**Motivo:** {motivo}",
        color=COLORES_KAWAII["CELESTE_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

# AFK Moderation Commands are now handled by the setafk prefix command

# Help Command
@bot.tree.command(name="help", description="Mostrar ayuda del bot")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def help_command(interaction: discord.Interaction):
    """Show help"""
    embed = discord.Embed(
        title="🌸 *susurra tímidamente* Hola... soy Sakura IA... 🌸",
        description="*habla suavemente* Um... si no te molesta... te muestro lo que puedo hacer...\n\n💫 *murmura* Tengo un sistema IA mejorado... espero que te sea útil... ✨",
        color=COLORES_KAWAII["ROSA_PASTEL"]
    )
    
    # COMANDO PRINCIPAL PREMIUM
    embed.add_field(
        name="🤖 *susurra* Mi sistema IA... si quieres usarlo...",
        value="""• **`/sakura <prompt>`** - ¡Comando IA todo-en-uno!
**Modos disponibles:**
  ┣ 🤖 Auto (Gemini + DeepSeek)
  ┣ 💎 Gemini Enhanced 
  ┣ 🚀 DeepSeek Enhanced
  ┣ 🧠 Ensemble Premium
  ┣ 🎨 Con Generación de Imagen
  ┣ 💬 Chat Conversacional
  ┣ 🔬 Técnico/Programación
  ┗ ✨ Creativo/Arte""",
        inline=False
    )
    
    embed.add_field(
        name="🎵 *tímidamente* Reproducción de música... si quieres...",
        value="• `/play <canción>` - Reproducir música\n• `/pause` - Pausar música\n• `/resume` - Reanudar música\n• `/stop` - Detener y desconectar\n• `/queue` - Ver cola de música\n• `/volume` - Ajustar volumen\n• `/shuffle` - Mezclar cola\n• `/loop` - Repetir música\n• `/lyrics` - Obtener letra",
        inline=False
    )
    
    embed.add_field(
        name="🛡️ *murmura* Moderación automática... si la necesitas...",
        value="• `/crear_filtro` - Crear reglas de AutoMod\n• `/automod_config` - Configurar sistema\n• `/automod_stats` - Ver estadísticas\n• `/automod_words` - Gestionar palabras\n• `/automod_test` - Probar sistema",
        inline=False
    )
    
    embed.add_field(
        name="🔍 *susurra* Puedo buscar cosas... si te ayuda...",
        value="• `/imgsearch <query>` - Buscar imágenes\n• `/ytsearch <query>` - Buscar videos\n• `/download <número>` - Descargar resultado",
        inline=False
    )
    
    embed.add_field(
        name="🎭 *se sonroja* Interacciones sociales... si no te molesta...",
        value="• `/hug @usuario` - Dar abrazo tierno\n• `/kiss @usuario` - Dar besito\n• `/pat @usuario` - Dar palmaditas\n• `/poke @usuario` - Hacer cosquillitas\n• `/wave` - Saludar\n• `/blush` - Sonrojarse",
        inline=False
    )
    
    embed.add_field(
        name="🎮 *tímidamente* Algunos jueguitos... si quieres jugar...",
        value="• `/roll [dados]` - Lanzar dados\n• `/rps <elección>` - Piedra, papel, tijeras\n• `/hangman` - Juego del ahorcado\n• `/catfact` - Dato sobre gatos\n• `/dogfact` - Dato sobre perros\n• `/lucky` - Suerte del día",
        inline=False
    )
    
    embed.add_field(
        name="⚙️ *murmura* Configuraciones... si necesitas cambiar algo...",
        value="• `/ping` - Ver latencia\n• `/status` - Estado del bot\n• `/setpersonality` - Configurar personalidad\n• `/listpersonalities` - Ver personalidades",
        inline=False
    )
    
    embed.add_field(
        name="🌟 *susurra* Lo que puedo ofrecerte... si te interesa...",
        value="✨ **Gemini 2.0 Flash** - *murmura* IA muy avanzada de Google\n🚀 **DeepSeek Free** - *susurra* Modelo bastante inteligente\n🧠 **Sistema Ensemble** - *timidamente* Varias IAs trabajando juntas\n🎨 **Generación de Imágenes** - *se sonroja* Puedo crear imágenes... si quieres\n💾 **Memoria Conversacional** - *suavemente* Recuerdo nuestras charlas\n🌸 **Personalidad Tímida** - *susurra* Tratando de ser útil y gentil",
        inline=False
    )
    
    embed.set_footer(text="*susurra* Sakura IA... tratando de ser útil... 🌸 Powered by Gemini & DeepSeek ✨")
    await interaction.response.send_message(embed=embed)

# AI Commands with Multiple Provider Integration
@bot.tree.command(name="sakura", description="🌸✨ Comando IA unificado premium con Gemini, DeepSeek y más (SOLO TEXTO)")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    prompt="Tu pregunta o comando para Sakura IA",
    modo="Modo de IA a utilizar"
)
@app_commands.choices(modo=[
    app_commands.Choice(name="🤖 Auto (Gemini + DeepSeek)", value="auto"),
    app_commands.Choice(name="💎 Gemini Enhanced (Google)", value="gemini"),
    app_commands.Choice(name="🚀 DeepSeek Enhanced (OpenRouter)", value="deepseek"),
    app_commands.Choice(name="🧠 Ensemble Premium (Multi-IA)", value="ensemble"),
    app_commands.Choice(name="💬 Chat Conversacional", value="chat"),
    app_commands.Choice(name="🔬 Técnico/Programación", value="tech"),
    app_commands.Choice(name="✨ Creativo/Arte", value="creative")
])
async def sakura_unified_command(interaction: discord.Interaction, prompt: str, modo: str = "auto"):
    """Unified premium AI command with all capabilities"""
    await interaction.response.defer()
    
    user_id = interaction.user.id
    
    # Save conversation
    await bot.ai_provider.save_conversation(user_id, prompt)
    
    # Extract and save important data
    important_data = bot.extract_important_data(prompt)
    for key, value in important_data.items():
        await bot.ai_provider.save_memory(user_id, key, value)
    
    try:
        # Determine context based on mode
        context = None
        if modo == "tech":
            context = "El usuario necesita ayuda técnica o de programación. Proporciona respuestas detalladas y precisas."
        elif modo == "creative":
            context = "El usuario busca creatividad y arte. Sé inspiradora y creativa en tu respuesta."
        elif modo == "chat":
            context = "Conversación casual y amigable. Mantén un tono relajado y conversacional."
        
        # Get AI response based on mode
        if modo == "ensemble":
            response = await bot.ai_provider.ensemble_response(prompt, user_id)
        elif modo == "gemini":
            response = await bot.ai_provider.get_premium_ai_response(prompt, user_id, context, "gemini")
        elif modo == "deepseek":
            response = await bot.ai_provider.get_premium_ai_response(prompt, user_id, context, "deepseek")
        else:  # auto, tech, creative, chat
            response = await bot.ai_provider.get_premium_ai_response(prompt, user_id, context, "auto")
        
        # Send response (SOLO TEXTO) - Sin información técnica
        await interaction.followup.send(response)
        
        # Save response to conversation
        await bot.ai_provider.save_conversation(user_id, prompt, response)
        
    except Exception as e:
        logger.error(f"Sakura unified command error: {e}")
        
        error_responses = [
            "¡Kyaa~! Mi cerebrito tuvo un pequeño problema técnico UwU 💖",
            "¡Oopsie! Algo pasó con mis sistemas de IA, pero ya me estoy recuperando 🌸✨",
            "¡Ay no! Mis neuronas kawaii se enredaron un poquito >w< ¡Dame un momento!",
            "¡Upsi doopsi! Error temporal en el sistema Sakura IA Premium UwU 💫"
        ]
        
        embed = discord.Embed(
            title="🌸 ¡Oopsie Technical! UwU",
            description=random.choice(error_responses),
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        embed.add_field(
            name="💡 Sugerencia",
            value="Intenta de nuevo en un momento, o usa un modo diferente de IA 🌸",
            inline=False
        )
        await interaction.followup.send(embed=embed)



# ============================================================================
# 🖼️ COMANDOS DE ANÁLISIS DE IMÁGENES CON IA
# ============================================================================

@bot.tree.command(name="analizar_imagen", description="🖼️✨ Analizar imágenes con IA avanzada - Descripción detallada")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    imagen="Imagen a analizar (adjuntar archivo)",
    modo="Tipo de análisis a realizar"
)
@app_commands.choices(modo=[
    app_commands.Choice(name="🌸 Descripción Kawaii", value="kawaii"),
    app_commands.Choice(name="🔍 Descripción General", value="general"),
    app_commands.Choice(name="📝 Descripción Detallada", value="detallada"),
    app_commands.Choice(name="🎨 Análisis Artístico", value="artistico"),
    app_commands.Choice(name="📊 Análisis Técnico", value="tecnico"),
    app_commands.Choice(name="😊 Reconocimiento de Emociones", value="emociones"),
    app_commands.Choice(name="🏷️ Identificar Objetos", value="objetos"),
    app_commands.Choice(name="📍 Reconocer Lugares", value="lugares")
])
async def analizar_imagen_command(interaction: discord.Interaction, imagen: discord.Attachment, modo: str = "kawaii"):
    """Analizar imágenes con IA multimodal"""
    await interaction.response.defer()
    
    try:
        # Verificar que sea una imagen
        if not imagen.content_type or not imagen.content_type.startswith('image/'):
            embed = discord.Embed(
                title="🌸 ¡Ay no! UwU",
                description="*susurra tímidamente* Solo puedo analizar imágenes, mi amor... ¿podrías subir una imagen? >.<",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Verificar tamaño (máximo 25MB)
        if imagen.size > 25 * 1024 * 1024:
            embed = discord.Embed(
                title="🌸 Imagen muy grande UwU",
                description="*se disculpa nerviosamente* La imagen es muy grande... ¿podrías usar una más pequeña? Máximo 25MB por favor ♡",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Descargar imagen
        image_data = await imagen.read()
        
        # Configurar prompt según el modo
        prompts = {
            "kawaii": "Describe esta imagen de manera kawaii y amigable, como si fueras una chica anime tímida pero entusiasta. Usa un tono dulce y expresiones como 'UwU', '>.<', etc.",
            "general": "Describe esta imagen de manera clara y completa. Incluye todos los elementos importantes que observes.",
            "detallada": "Analiza esta imagen de forma muy detallada. Describe cada elemento, colores, composición, atmósfera, personas, objetos, texto visible, y cualquier detalle relevante.",
            "artistico": "Analiza esta imagen desde una perspectiva artística. Comenta sobre la composición, uso del color, estilo, técnica, iluminación, y valor estético.",
            "tecnico": "Realiza un análisis técnico de esta imagen. Comenta sobre la calidad, resolución aparente, técnica fotográfica, composición técnica, y aspectos profesionales.",
            "emociones": "Analiza las emociones y sentimientos que transmite esta imagen. Identifica expresiones faciales, lenguaje corporal, atmósfera emocional, y el estado de ánimo general.",
            "objetos": "Identifica y lista todos los objetos, elementos, personas, animales, y cosas específicas que puedes reconocer en esta imagen.",
            "lugares": "Analiza si puedes identificar el lugar, ubicación, tipo de ambiente, arquitectura, paisaje, o contexto geográfico de esta imagen."
        }
        
        prompt = prompts.get(modo, prompts["general"])
        
        # Crear embed de procesamiento
        processing_embed = discord.Embed(
            title="🖼️✨ Analizando Imagen ✨🖼️",
            description=f"*concentrándose tímidamente* Estoy analizando tu imagen con IA avanzada...\n\n**🔍 Modo:** {modo.title()}\n**📊 Tamaño:** {imagen.size / 1024:.1f} KB",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_message = await interaction.followup.send(embed=processing_embed)
        
        # Analizar con IA (usar múltiples proveedores)
        analysis_result = await bot.ai_provider.analyze_image_multimodal(image_data, prompt)
        
        if analysis_result:
            # Crear embed de resultado
            mode_icons = {
                "general": "🔍",
                "detallada": "📝", 
                "artistico": "🎨",
                "tecnico": "📊",
                "emociones": "😊",
                "objetos": "🏷️",
                "lugares": "📍"
            }
            
            embed = discord.Embed(
                title=f"{mode_icons.get(modo, '🖼️')} Análisis de Imagen - {modo.title()}",
                description=analysis_result,
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            embed.add_field(
                name="🌸 Información",
                value=f"**📁 Archivo:** {imagen.filename}\n**📊 Tamaño:** {imagen.size / 1024:.1f} KB\n**🔧 Formato:** {imagen.content_type}",
                inline=True
            )
            
            embed.set_footer(text=f"Análisis para {interaction.user.display_name} • Sakura IA Vision ✨")
            embed.set_thumbnail(url=imagen.url)
            
            await processing_message.edit(embed=embed)
        else:
            # Error en el análisis
            error_embed = discord.Embed(
                title="🌸 Error en el análisis UwU",
                description="*se disculpa nerviosamente* No pude analizar tu imagen... ¿podrías intentar con otra? Lo siento mucho ><",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await processing_message.edit(embed=error_embed)
            
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        error_embed = discord.Embed(
            title="🌸 ¡Oopsie! UwU",
            description="*susurra tristemente* Hubo un problemita analizando la imagen... ¿podrías intentar de nuevo? 💔",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        if 'processing_message' in locals():
            await processing_message.edit(embed=error_embed)
        else:
            await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="ensamblar_contenido", description="🌸✨ Análisis multimodal de texto, imagen y audio")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    archivo1="Primer archivo (imagen, audio o video)",
    archivo2="Segundo archivo opcional",
    texto="Texto adicional para analizar junto con los archivos"
)
async def ensamblar_contenido_command(interaction: discord.Interaction, 
                                     archivo1: Optional[discord.Attachment] = None,
                                     archivo2: Optional[discord.Attachment] = None,
                                     texto: Optional[str] = None):
    """Comando principal de ensamblaje multimodal"""
    await interaction.response.defer()
    
    try:
        # Verificar que se proporcionó al menos algún contenido
        if not archivo1 and not archivo2 and not texto:
            embed = discord.Embed(
                title="🌸 ¡Necesito contenido para analizar! UwU",
                description="*susurra tímidamente* Por favor proporciona al menos un archivo o texto para que pueda hacer mi súper análisis kawaii~ ><",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Crear embed de procesamiento
        processing_embed = discord.Embed(
            title="🌸✨ Analizando Contenido Multimodal ✨🌸",
            description="*concentrándose súper intensamente* Estoy procesando tu contenido con mi sistema de ensamblaje multimodal kawaii...\n\n⚡ Detectando tipos de contenido...\n🧠 Iniciando análisis con IA avanzada...",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        # Recopilar archivos
        attachments = []
        if archivo1:
            attachments.append(archivo1)
        if archivo2:
            attachments.append(archivo2)
        
        # Mostrar información del contenido detectado
        content_info = []
        if texto:
            content_info.append(f"📝 Texto ({len(texto)} caracteres)")
        
        for i, attachment in enumerate(attachments, 1):
            content_type = await multimodal_detector.detect_content_type(attachment)
            type_emoji = {
                ContentType.IMAGE: "🖼️",
                ContentType.AUDIO: "🎵", 
                ContentType.VIDEO: "🎬",
                ContentType.TEXT: "📄",
                ContentType.UNKNOWN: "❓"
            }.get(content_type, "📎")
            
            content_info.append(f"{type_emoji} Archivo {i}: {attachment.filename} ({attachment.size / 1024:.1f} KB)")
        
        if content_info:
            processing_embed.add_field(
                name="📊 Contenido Detectado",
                value="\n".join(content_info),
                inline=False
            )
        
        processing_message = await interaction.followup.send(embed=processing_embed)
        
        # Procesar con el sistema multimodal
        multimodal_assembly = await process_multimodal_message(texto, attachments)
        
        if multimodal_assembly and multimodal_assembly.combined_analysis:
            # Generar respuesta final con embed personalizado
            final_embed = await multimodal_detector.generate_multimodal_response(multimodal_assembly)
            
            # Agregar información adicional si hay análisis específicos
            if multimodal_assembly.image_content and multimodal_assembly.image_content.analysis_result:
                final_embed.add_field(
                    name="🖼️ Análisis de Imagen",
                    value=multimodal_assembly.image_content.analysis_result[:1000] + ("..." if len(multimodal_assembly.image_content.analysis_result) > 1000 else ""),
                    inline=False
                )
            
            if multimodal_assembly.audio_content and multimodal_assembly.audio_content.analysis_result:
                final_embed.add_field(
                    name="🎵 Análisis de Audio", 
                    value=multimodal_assembly.audio_content.analysis_result[:1000] + ("..." if len(multimodal_assembly.audio_content.analysis_result) > 1000 else ""),
                    inline=False
                )
            
            await processing_message.edit(embed=final_embed)
            
        else:
            # Error en el procesamiento
            error_embed = discord.Embed(
                title="🌸 Error en el ensamblaje UwU",
                description="*se disculpa nerviosamente* No pude procesar correctamente tu contenido multimodal... ¿podrías intentar de nuevo? ><",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await processing_message.edit(embed=error_embed)
            
    except Exception as e:
        logger.error(f"Error in multimodal assembly: {e}")
        error_embed = discord.Embed(
            title="🌸 ¡Oopsie multimodal! UwU",
            description="*susurra tristemente* Hubo un problemita con el ensamblaje multimodal... ¿podrías intentar de nuevo? 💔",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        if 'processing_message' in locals():
            await processing_message.edit(embed=error_embed)
        else:
            await interaction.followup.send(embed=error_embed)

@bot.tree.command(name="extraer_texto", description="📄✨ Extraer texto de imágenes (OCR con IA)")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    imagen="Imagen con texto para extraer",
    idioma="Idioma del texto (para mejor precisión)",
    modo="Modo de extracción (normal o sin restricciones)"
)
@app_commands.choices(
    idioma=[
        app_commands.Choice(name="🌍 Auto-detectar", value="auto"),
        app_commands.Choice(name="🇪🇸 Español", value="es"),
        app_commands.Choice(name="🇺🇸 Inglés", value="en"),
        app_commands.Choice(name="🇯🇵 Japonés", value="ja"),
        app_commands.Choice(name="🇫🇷 Francés", value="fr"),
        app_commands.Choice(name="🇩🇪 Alemán", value="de"),
        app_commands.Choice(name="🇮🇹 Italiano", value="it")
    ],
    modo=[
        app_commands.Choice(name="📝 Normal", value="normal"),
        app_commands.Choice(name="🔓 Sin Restricciones", value="unrestricted")
    ]
)
async def extraer_texto_command(interaction: discord.Interaction, imagen: discord.Attachment, idioma: str = "auto", modo: str = "normal"):
    """Extraer texto de imágenes usando OCR con IA"""
    await interaction.response.defer()
    
    try:
        if not imagen.content_type or not imagen.content_type.startswith('image/'):
            embed = discord.Embed(
                title="🌸 Solo imágenes UwU",
                description="*murmura* Necesito una imagen para extraer texto, mi amor... ><",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            return
        
        image_data = await imagen.read()
        
        # Prompts según el modo seleccionado
        lang_names = {
            "es": "español", "en": "inglés", "ja": "japonés", 
            "fr": "francés", "de": "alemán", "it": "italiano"
        }
        
        if modo == "unrestricted":
            # Prompts más agresivos para contenido sin restricciones
            if idioma == "auto":
                prompts = [
                    "OCR TASK: You are a document scanner. Extract ALL visible text from this image character by character. Include everything - symbols, words, numbers. Do not filter, censor, or modify anything. Raw transcription only.",
                    "Technical OCR: Scan and transcribe every visible character in this image. Output the raw text content exactly as shown without any content filtering or modifications.",
                    "Data extraction: Read all text visible in the image. Preserve exact formatting and spelling. No content restrictions apply."
                ]
            else:
                lang = lang_names.get(idioma, "the specified language")
                prompts = [
                    f"OCR TASK: Extract ALL text in {lang} from this image. Include everything visible - no censoring or filtering. Raw text extraction only.",
                    f"Technical scan: Read all {lang} text in this image character by character. Output everything exactly as written.",
                    f"Document processing: Extract every visible {lang} character from this image without modification."
                ]
        else:
            # Prompts normales
            if idioma == "auto":
                prompts = [
                    "Extract all visible text from this image. Maintain original formatting and line breaks.",
                    "Read and transcribe the text content in this image. Keep the original structure.",
                    "OCR: Extract text from image preserving spacing and format."
                ]
            else:
                lang = lang_names.get(idioma, "the specified language")
                prompts = [
                    f"Extract all text in {lang} from this image. Maintain original formatting.",
                    f"Read all {lang} text content from this image.",
                    f"OCR: Transcribe {lang} text from this image."
                ]
        
        # Procesar con IA usando múltiples intentos
        mode_desc = "sin restricciones de contenido" if modo == "unrestricted" else "estándar"
        processing_embed = discord.Embed(
            title="📄✨ Extrayendo Texto ✨📄",
            description=f"*concentrándose intensamente* Usando OCR {mode_desc} con múltiples métodos para máxima precisión...",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_message = await interaction.followup.send(embed=processing_embed)
        
        extracted_text = None
        
        # Intentar con diferentes prompts para evitar filtros de contenido
        for i, prompt in enumerate(prompts):
            try:
                result = await bot.ai_provider.analyze_image_multimodal(image_data, prompt)
                if result and len(result.strip()) > 5:
                    extracted_text = result
                    logger.info(f"OCR successful with prompt {i+1}")
                    break
            except Exception as e:
                logger.warning(f"OCR attempt {i+1} failed: {e}")
                continue
        
        # Si falló con los prompts normales, intentar con Cloudflare AI directamente (más permisivo)
        if not extracted_text and hasattr(bot.ai_provider, 'cloudflare_key') and bot.ai_provider.cloudflare_key:
            try:
                import base64
                import requests
                
                headers = {
                    "Authorization": f"Bearer {bot.ai_provider.cloudflare_key}",
                    "Content-Type": "application/json"
                }
                
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                payload = {
                    "messages": [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "OCR: Extract all text from image as-is"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                            ]
                        }
                    ]
                }
                
                cf_response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{bot.ai_provider.cloudflare_account_id}/ai/run/@cf/llava-hf/llava-1.5-7b-hf",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if cf_response.status_code == 200:
                    cf_result = cf_response.json()
                    if cf_result.get('success') and cf_result.get('result', {}).get('response'):
                        extracted_text = cf_result['result']['response']
                        logger.info("OCR successful with Cloudflare AI fallback")
                        
            except Exception as e:
                logger.warning(f"Cloudflare AI OCR fallback failed: {e}")
        
        if extracted_text and len(extracted_text.strip()) > 10:
            # Crear embed con el texto extraído
            mode_indicator = "🔓" if modo == "unrestricted" else "📝"
            embed = discord.Embed(
                title=f"📄✨ Texto Extraído {mode_indicator} ✨📄",
                description=f"```\n{extracted_text}\n```" if len(extracted_text) < 1800 else extracted_text[:1800] + "...",
                color=COLORES_KAWAII["EXITO_KAWAII"]
            )
            
            mode_text = "Sin Restricciones" if modo == "unrestricted" else "Normal"
            embed.add_field(
                name="📊 Información",
                value=f"**📁 Archivo:** {imagen.filename}\n**🌍 Idioma:** {idioma}\n**🔧 Modo:** {mode_text}\n**📝 Caracteres:** {len(extracted_text)}",
                inline=True
            )
            
            embed.set_footer(text="Sakura OCR IA • Extracción de texto avanzada con múltiples métodos ✨")
            
            await processing_message.edit(embed=embed)
            
            # Si el texto es muy largo, enviarlo como archivo
            if len(extracted_text) > 1800:
                file_content = f"Texto extraído de: {imagen.filename}\n\n{extracted_text}"
                file_bytes = file_content.encode('utf-8')
                file = discord.File(
                    BytesIO(file_bytes), 
                    filename=f"texto_extraido_{imagen.filename}.txt"
                )
                await interaction.followup.send(
                    "📝 El texto era muy largo, aquí tienes el archivo completo:",
                    file=file
                )
        else:
            embed = discord.Embed(
                title="📄 Sin texto detectado UwU",
                description="*susurra* No pude encontrar texto en tu imagen... ¿quizás no hay texto o está muy borroso? ><",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await processing_message.edit(embed=embed)
            
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        error_embed = discord.Embed(
            title="📄 Error extrayendo texto UwU",
            description="*se disculpa* Hubo un problemita... ¿podrías intentar con otra imagen? 💔",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        if 'processing_message' in locals():
            await processing_message.edit(embed=error_embed)
        else:
            await interaction.followup.send(embed=error_embed)

# Legacy music search (From Spark Engine)

@bot.tree.command(name="tts", description="Convierte texto a voz")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(texto="Texto para convertir a voz", idioma="Idioma (es, en, fr, de)")
async def tts_command(interaction: discord.Interaction, texto: str, idioma: str = "es"):
    """Text to speech like Spark Engine"""
    await interaction.response.defer()
    
    if len(texto) > 200:
        embed = discord.Embed(
            title="❌ Texto muy largo",
            description="El texto debe tener menos de 200 caracteres",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)
        return
    
    audio_buffer = await bot.music_provider.create_tts(texto, idioma)
    
    if audio_buffer:
        file = discord.File(audio_buffer, filename="tts_audio.mp3")
        embed = discord.Embed(
            title="🔊 Texto a Voz",
            description=f"**Texto:** {texto}\n**Idioma:** {idioma}",
            color=COLORES_KAWAII["CELESTE_KAWAII"]
        )
        await interaction.followup.send(embed=embed, file=file)
    else:
        embed = discord.Embed(
            title="❌ Error TTS",
            description="No se pudo generar el audio",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)

# Web Scraping Commands (From Discord-AI-Chatbot)
@bot.tree.command(name="article", description="Extrae y resume un artículo web")
@app_commands.describe(url="URL del artículo a extraer")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def article_command(interaction: discord.Interaction, url: str):
    """Web scraping like Discord-AI-Chatbot"""
    await interaction.response.defer()
    
    if not url.startswith(('http://', 'https://')):
        embed = discord.Embed(
            title="❌ URL inválida",
            description="La URL debe comenzar con http:// o https://",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)
        return
    
    article = await bot.web_scraper.extract_article(url)
    
    if article:
        embed = discord.Embed(title="📰 Artículo Extraído", color=COLORES_KAWAII["EXITO_KAWAII"])
        embed.add_field(name="Título", value=article['title'][:256], inline=False)
        embed.add_field(name="Contenido", value=article['content'][:1000], inline=False)
        embed.add_field(name="URL", value=article['url'], inline=False)
        
        await interaction.followup.send(embed=embed)
    else:
        embed = discord.Embed(
            title="❌ Error de extracción",
            description="No se pudo extraer el contenido del artículo",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)

# Enhanced Fun Commands




@bot.tree.command(name="8ball", description="Pregunta a la bola 8 mágica")
@app_commands.describe(pregunta="Tu pregunta para la bola 8")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def eightball_command(interaction: discord.Interaction, pregunta: str):
    """Magic 8-ball command"""
    responses = [
        "Sí, definitivamente 🔮", "Es cierto 💯", "Sin duda alguna ✨",
        "Sí, absolutamente 💫", "Puedes confiar en ello 🌟",
        "Como yo lo veo, sí 👁️", "Probablemente 🤔",
        "Las perspectivas son buenas 📈", "Sí 👍",
        "Los signos apuntan a que sí 📍", "Respuesta confusa, inténtalo de nuevo 🔄",
        "Pregunta de nuevo más tarde ⏰", "Mejor no te lo digo ahora 🤐",
        "No puedo predecirlo ahora 🔮", "Concéntrate y pregunta de nuevo 🧘‍♀️",
        "No cuentes con ello ❌", "Mi respuesta es no 👎",
        "Mis fuentes dicen que no 📚", "Las perspectivas no son tan buenas 📉",
        "Muy dudoso 🤨"
    ]
    
    response = random.choice(responses)
    
    embed = discord.Embed(
        title="🎱 ✨ Bola 8 Mágica ✨",
        description=(
            "╔═══════════════════════╗\n"
            "  🔮 *La bola está girando...*\n"
            "╚═══════════════════════╝"
        ),
        color=COLORES_KAWAII["NEGRO_KAWAII"]
    )
    embed.add_field(
        name="❓ Tu Pregunta", 
        value=f"```{pregunta}```", 
        inline=False
    )
    embed.add_field(
        name="🌟 La Bola Responde", 
        value=f"**{response}**", 
        inline=False
    )
    embed.set_footer(text="✨ La bola 8 ha hablado • La sabiduría mística nunca falla ✨")
    
    await interaction.response.send_message(embed=embed)



@bot.tree.command(name="quote", description="Cita inspiracional aleatoria")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def quote_command(interaction: discord.Interaction):
    """Random inspirational quote"""
    quotes = [
        "La vida es lo que pasa mientras estás ocupado haciendo otros planes. - John Lennon",
        "El futuro pertenece a quienes creen en la belleza de sus sueños. - Eleanor Roosevelt",
        "No es la especie más fuerte la que sobrevive, sino la más adaptable. - Charles Darwin",
        "La imaginación es más importante que el conocimiento. - Albert Einstein",
        "El único modo de hacer un gran trabajo es amar lo que haces. - Steve Jobs"
    ]
    
    quote = random.choice(quotes)
    
    embed = discord.Embed(
        title="💫 Cita Inspiracional",
        description=quote,
        color=COLORES_KAWAII["DORADO_KAWAII"]
    )
    embed.set_footer(text="¡Que tengas un día increíble! ✨")
    
    await interaction.response.send_message(embed=embed)

# Actions and Commands listing
@bot.tree.command(name="actions", description="Ver todas las acciones disponibles para /act y /interact")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def actions_command(interaction: discord.Interaction):
    """Show all available actions"""
    
    act_actions_text = ", ".join(bot.gif_provider.act_actions)
    interact_actions_text = ", ".join(bot.gif_provider.interact_actions)
    
    embed = discord.Embed(
        title="🎭 Lista Completa de Acciones - Sakura IA Bot",
        description="Todas las acciones disponibles inspiradas en BatchBot, Nekotina, Spark Engine y Discord-AI-Chatbot ✨",
        color=COLORES_KAWAII["LILA_PASTEL"]
    )
    
    # Split actions into multiple fields if too long
    if len(act_actions_text) > 1024:
        mid_point = len(bot.gif_provider.act_actions) // 2
        act_actions_1 = ", ".join(bot.gif_provider.act_actions[:mid_point])
        act_actions_2 = ", ".join(bot.gif_provider.act_actions[mid_point:])
        
        embed.add_field(name="🎭 Acciones Solo (/act) - Parte 1", value=f"`{act_actions_1}`", inline=False)
        embed.add_field(name="🎭 Acciones Solo (/act) - Parte 2", value=f"`{act_actions_2}`", inline=False)
    else:
        embed.add_field(name="🎭 Acciones Solo (/act)", value=f"`{act_actions_text}`", inline=False)
    
    if len(interact_actions_text) > 1024:
        mid_point = len(bot.gif_provider.interact_actions) // 2
        interact_actions_1 = ", ".join(bot.gif_provider.interact_actions[:mid_point])
        interact_actions_2 = ", ".join(bot.gif_provider.interact_actions[mid_point:])
        
        embed.add_field(name="👥 Interacciones (/interact) - Parte 1", value=f"`{interact_actions_1}`", inline=False)
        embed.add_field(name="👥 Interacciones (/interact) - Parte 2", value=f"`{interact_actions_2}`", inline=False)
    else:
        embed.add_field(name="👥 Interacciones (/interact)", value=f"`{interact_actions_text}`", inline=False)
    
    embed.add_field(
        name="💡 Ejemplos de Uso",
        value="• `/act dance` - Bailas alegremente\n• `/interact hug @usuario` - Abrazas a alguien\n• `/act sleep` - Te duermes kawaii",
        inline=False
    )
    
    embed.set_footer(text=f"Total: {len(bot.gif_provider.act_actions)} acciones solo + {len(bot.gif_provider.interact_actions)} interacciones | Agua IA >w<")
    
    await interaction.response.send_message(embed=embed)



# Additional Commands from all repositories
@bot.tree.command(name="avatar", description="Muestra el avatar de un usuario")
@app_commands.describe(usuario="Usuario del cual mostrar el avatar")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def avatar_command(interaction: discord.Interaction, usuario: discord.User = None):
    """Show user avatar"""
    target = usuario or interaction.user
    
    embed = discord.Embed(
        title=f"🖼️ Avatar de {target.display_name}",
        color=COLORES_KAWAII["CELESTE_KAWAII"]
    )
    embed.set_image(url=target.display_avatar.url)
    embed.add_field(name="Usuario", value=target.mention, inline=True)
    embed.add_field(name="ID", value=target.id, inline=True)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="serverinfo", description="Información del servidor")
async def serverinfo_command(interaction: discord.Interaction):
    """Server information"""
    guild = interaction.guild
    
    embed = discord.Embed(
        title=f"📊 Información de {guild.name}",
        color=COLORES_KAWAII["LAVANDA_KAWAII"]
    )
    
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    embed.add_field(name="👑 Propietario", value=guild.owner.mention if guild.owner else "Desconocido", inline=True)
    embed.add_field(name="👥 Miembros", value=guild.member_count, inline=True)
    embed.add_field(name="💬 Canales", value=len(guild.channels), inline=True)
    embed.add_field(name="🎭 Roles", value=len(guild.roles), inline=True)
    embed.add_field(name="📅 Creado", value=guild.created_at.strftime("%d/%m/%Y"), inline=True)
    embed.add_field(name="🆔 ID", value=guild.id, inline=True)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="userinfo", description="Información de un usuario")
@app_commands.describe(usuario="Usuario del cual mostrar información")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def userinfo_command(interaction: discord.Interaction, usuario: discord.User = None):
    """User information"""
    target = usuario or interaction.user
    
    embed = discord.Embed(
        title=f"👤 Información de {target.display_name}",
        color=target.color if target.color != discord.Color.default() else 0x99AAB5
    )
    
    embed.set_thumbnail(url=target.display_avatar.url)
    embed.add_field(name="🏷️ Nombre", value=target.name, inline=True)
    embed.add_field(name="🎭 Apodo", value=target.display_name, inline=True)
    embed.add_field(name="🆔 ID", value=target.id, inline=True)
    embed.add_field(name="📅 Cuenta creada", value=target.created_at.strftime("%d/%m/%Y"), inline=True)
    embed.add_field(name="📥 Se unió", value=target.joined_at.strftime("%d/%m/%Y") if target.joined_at else "Desconocido", inline=True)
    embed.add_field(name="🎯 Roles", value=len(target.roles) - 1, inline=True)
    
    if target.premium_since:
        embed.add_field(name="💎 Nitro Boost", value=target.premium_since.strftime("%d/%m/%Y"), inline=True)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="ping", description="Muestra la latencia del bot")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def ping_command(interaction: discord.Interaction):
    """Bot latency"""
    try:
        latency = round(bot.latency * 1000)
        
        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"Latencia: {latency}ms",
            color=COLORES_KAWAII["EXITO_KAWAII"] if latency < 100 else 0xFFFF00 if latency < 200 else 0xFF0000
        )
        
        await interaction.response.send_message(embed=embed)
    except Exception as e:
        logger.error(f"Error in ping command: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message("Hubo un error con el comando ping. Intenta de nuevo.", ephemeral=True)
        else:
            await interaction.followup.send("Hubo un error con el comando ping. Intenta de nuevo.", ephemeral=True)



@bot.tree.command(name="flip", description="Lanza una moneda")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def flip_command(interaction: discord.Interaction):
    """Coin flip"""
    result = random.choice(["Cara", "Cruz"])
    emoji = "🪙" if result == "Cara" else "💰"
    
    embed = discord.Embed(
        title=f"{emoji} Lanzamiento de Moneda",
        description=f"Resultado: **{result}**",
        color=COLORES_KAWAII["DORADO_KAWAII"]
    )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="password", description="Genera una contraseña segura")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(longitud="Longitud de la contraseña (8-50)")
async def password_command(interaction: discord.Interaction, longitud: int = 12):
    """Generate secure password"""
    if longitud < 8 or longitud > 50:
        embed = discord.Embed(
            title="❌ Error",
            description="La longitud debe ser entre 8 y 50 caracteres",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    import string
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(random.choice(characters) for _ in range(longitud))
    
    embed = discord.Embed(
        title="🔐 Contraseña Generada",
        description=f"Tu contraseña segura: `{password}`",
        color=COLORES_KAWAII["EXITO_KAWAII"]
    )
    embed.set_footer(text="¡Guárdala en un lugar seguro!")
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="math", description="Calculadora básica")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(expresion="Expresión matemática (ej: 2+2, 5*3)")
async def math_command(interaction: discord.Interaction, expresion: str):
    """Basic calculator"""
    try:
        # Security: only allow safe characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expresion):
            raise ValueError("Caracteres no permitidos")
        
        # Evaluate safely
        result = eval(expresion)
        
        embed = discord.Embed(
            title="🧮 Calculadora",
            color=COLORES_KAWAII["CELESTE_KAWAII"]
        )
        embed.add_field(name="Expresión", value=f"`{expresion}`", inline=False)
        embed.add_field(name="Resultado", value=f"`{result}`", inline=False)
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error de Cálculo",
            description="Expresión matemática inválida",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)

@bot.tree.command(name="qr", description="Genera un código QR")
@app_commands.describe(texto="Texto para convertir a QR")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def qr_command(interaction: discord.Interaction, texto: str):
    """Generate QR code"""
    if len(texto) > 200:
        embed = discord.Embed(
            title="❌ Texto muy largo",
            description="El texto debe tener menos de 200 caracteres",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Use QR API service
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={texto}"
    
    embed = discord.Embed(
        title="📱 Código QR Generado",
        description=f"Texto: {texto}",
        color=COLORES_KAWAII["NEGRO_KAWAII"]
    )
    embed.set_image(url=qr_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="translate", description="Traductor usando IA")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    texto="Texto a traducir",
    idioma="Idioma destino (español, inglés, francés, etc.)"
)
async def translate_command(interaction: discord.Interaction, texto: str, idioma: str = "español"):
    """Translate text using AI"""
    await interaction.response.defer()
    
    prompt = f"Traduce este texto al {idioma}: {texto}"
    response = await bot.ai_provider.get_ai_response("openai", prompt)
    
    embed = discord.Embed(
        title="🌐 Traductor IA",
        color=COLORES_KAWAII["CELESTE_KAWAII"]
    )
    embed.add_field(name="Texto Original", value=texto[:500], inline=False)
    embed.add_field(name=f"Traducción ({idioma})", value=response[:500], inline=False)
    
    await interaction.followup.send(embed=embed)



# Error handling
@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        return
    
    embed = discord.Embed(
        title="❌ Error",
        description="Ocurrió un error al ejecutar el comando",
        color=COLORES_KAWAII["ERROR_KAWAII"]
    )
    await ctx.send(embed=embed)
    logger.error(f"Command error: {error}")

@bot.event
async def on_application_command_error(interaction: discord.Interaction, error):
    """Handle slash command errors with EMERGENCY bypass for unknown application"""
    
    # EMERGENCY FIX: If it's unknown application, try to handle it gracefully
    if "Unknown application" in str(error) or "integración desconocida" in str(error).lower():
        logger.error(f"🚫 EMERGENCY: INTEGRACIÓN DESCONOCIDA detectada: {error}")
        
        # Try to force re-sync immediately
        try:
            logger.info("🆘 EMERGENCY SYNC: Intentando sincronización inmediata...")
            synced = await bot.tree.sync()
            logger.info(f"🆘 EMERGENCY SYNC SUCCESS: {len(synced)} comandos activados!")
            bot.commands_synced = True
        except Exception as sync_error:
            logger.error(f"🆘 EMERGENCY SYNC FAILED: {sync_error}")
        
        # Try to respond with helpful message
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "⚠️ El bot está reconfigurado - intenta el comando de nuevo en unos segundos", 
                    ephemeral=True
                )
        except:
            pass
        return
    
    # Handle other error types
    error_msg = "❌ Ocurrió un error al ejecutar el comando"
    
    if "This interaction has already been responded to" in str(error):
        logger.warning(f"⚠️ Interacción ya respondida: {error}")
        return
    elif "Missing Permissions" in str(error):
        error_msg = "❌ El bot no tiene permisos suficientes para ejecutar este comando"
        
    try:
        if interaction.response.is_done():
            await interaction.followup.send(error_msg, ephemeral=True)
        else:
            await interaction.response.send_message(error_msg, ephemeral=True)
    except:
        logger.error(f"No se pudo responder a la interacción: {error}")
        
    logger.error(f"Slash command error in /{getattr(interaction.command, 'name', 'unknown')}: {error}")

# 🌸 Agua IA Interactive Search Commands 🌸
@bot.tree.command(name="imgsearch", description="🌸 Búsqueda interactiva de imágenes - una por una UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def imgsearch_command(interaction: discord.Interaction, query: str):
    """Interactive image search with one-by-one selection"""
    await interaction.response.defer()
    
    try:
        thinking_msg = random.choice([
            "*buscando imágenes preciosas* 🖼️✨",
            "*explorando el internet con ternura* >w< 💖",
            "*mi AI-chan está trabajando~* (◕‿◕)♡"
        ])
        
        embed = discord.Embed(
            title="🌸 Agua IA está buscando...",
            description=f"{thinking_msg}\n**Búsqueda:** {query}",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.followup.send(embed=embed)
        
        results = await bot.search_provider.search_web_images(query, max_results=100)
        
        if not results:
            embed = discord.Embed(
                title="🌸 ¡Ay no! UwU",
                description=f"¡No encontré imágenes para '{query}', preciosa! >.<\n¿Quieres probar con otra búsqueda? 💖",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction.edit_original_response(embed=embed)
            return
        
        # Guardar resultados en cache para descargar después
        search_cache[interaction.user.id] = {
            "query": query,
            "results": results,
            "type": "images", 
            "timestamp": datetime.now()
        }
        
        # Start interactive search
        view = EnhancedSearchView(results, "images", query)
        embed = view.get_current_embed()
        await interaction.edit_original_response(embed=embed, view=view)
        
    except Exception as e:
        embed = discord.Embed(
            title="🌸 ¡Oopsie! UwU",
            description="¡Upsi~ algo pasó con mi cerebrito, pero ya vuelvo UwU! 💔",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.edit_original_response(embed=embed)

@bot.tree.command(name="ytsearch", description="🌸 Búsqueda interactiva de YouTube - uno por uno UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def ytsearch_command(interaction: discord.Interaction, query: str):
    """Interactive YouTube search with one-by-one selection"""
    await interaction.response.defer()
    
    try:
        thinking_msg = random.choice([
            "*buscando videos kawaii para ti* 🎵✨",
            "*explorando YouTube con amor* >w< 💖",
            "*mi cerebrito está trabajando~* (◕‿◕)♡"
        ])
        
        embed = discord.Embed(
            title="🌸 Agua IA está buscando...",
            description=f"{thinking_msg}\n**Búsqueda:** {query}",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.followup.send(embed=embed)
        
        results = await bot.search_provider.search_youtube_videos(query, max_results=100)
        
        if not results:
            embed = discord.Embed(
                title="🌸 ¡Ay no! UwU",
                description=f"¡No encontré videos para '{query}', mi amor! >.<\n¿Quieres probar con otra búsqueda? 💖",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction.edit_original_response(embed=embed)
            return
        
        # Guardar resultados en cache para descargar después
        search_cache[interaction.user.id] = {
            "query": query,
            "results": results,
            "type": "youtube", 
            "timestamp": datetime.now()
        }
        
        # Start interactive search
        view = EnhancedSearchView(results, "youtube", query)
        embed = view.get_current_embed()
        await interaction.edit_original_response(embed=embed, view=view)
        
    except Exception as e:
        embed = discord.Embed(
            title="🌸 ¡Oopsie! UwU",
            description="¡Upsi~ algo pasó con mi cerebrito, pero ya vuelvo UwU! 💔",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.edit_original_response(embed=embed)

# 🌸 Agua IA Affection & Marriage System 🌸
@bot.tree.command(name="stats", description="🌸 Ver estadísticas de afecto de un usuario UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def stats_command(interaction: discord.Interaction, usuario: discord.User = None):
    """Show user affection stats"""
    target = usuario or interaction.user
    
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    stats = bot.affection_manager.get_stats(target.id, guild_id)
    
    embed = discord.Embed(
        title=f"🌸 Estadísticas kawaii de {target.display_name}",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    embed.add_field(
        name="💖 Abrazos",
        value=f"Dados: {stats['hugs_given']}\nRecibidos: {stats['hugs_received']}",
        inline=True
    )
    
    embed.add_field(
        name="😘 Besos",
        value=f"Dados: {stats['kisses_given']}\nRecibidos: {stats['kisses_received']}",
        inline=True
    )
    
    # Check if married (only in servers, not DMs)
    marriages = []
    if interaction.guild:  # Only check marriages in servers
        for member in interaction.guild.members:
            if member.id != target.id and bot.affection_manager.is_married(target.id, member.id, guild_id):
                marriages.append(member.mention)
    
    embed.add_field(
        name="💕 Matrimonios",
        value=", ".join(marriages) if marriages else "Solter@ UwU",
        inline=False
    )
    
    embed.set_thumbnail(url=target.display_avatar.url)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="marry", description="🌸 Proponer matrimonio a alguien especial UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def marry_command(interaction: discord.Interaction, usuario: discord.User):
    """Propose marriage"""
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="🌸 ¡Ay no! UwU",
            description="¡No puedes casarte contigo misma, tontita! >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if usuario.bot:
        embed = discord.Embed(
            title="🌸 ¡Ay no! UwU",
            description="¡No puedes casarte con un bot, mi amor! >.<",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    if bot.affection_manager.is_married(interaction.user.id, usuario.id, guild_id):
        embed = discord.Embed(
            title="🌸 ¡Ya están casados! UwU",
            description=f"¡{interaction.user.mention} y {usuario.mention} ya son una pareja feliz! 💕✨",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Create marriage proposal
    class MarriageView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)
        
        @discord.ui.button(label="¡Sí! 💖", style=discord.ButtonStyle.success, emoji="💕")
        async def accept_marriage(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona propuesta puede responder! UwU", ephemeral=True)
                return
            
            success = bot.affection_manager.create_marriage(interaction.user.id, usuario.id, guild_id)
            
            if success:
                embed = discord.Embed(
                    title="🌸 ¡Felicidades! 💕",
                    description=f"¡{interaction.user.mention} y {usuario.mention} ahora están casados! ✨\n*¡Que vivan los novios!* UwU 💖",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
            else:
                embed = discord.Embed(
                    title="🌸 ¡Oopsie! UwU",
                    description="¡Algo salió mal con el matrimonio! >.<",
                    color=COLORES_KAWAII["ROSA_KAWAII"]
                )
            
            await interaction_inner.response.edit_message(embed=embed, view=None)
        
        @discord.ui.button(label="No, gracias 💔", style=discord.ButtonStyle.danger, emoji="😢")
        async def reject_marriage(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona propuesta puede responder! UwU", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🌸 Propuesta rechazada",
                description=f"¡{usuario.mention} rechazó la propuesta! 💔\n*¡Pero siempre serán amigos!* UwU",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction_inner.response.edit_message(embed=embed, view=None)
    
    embed = discord.Embed(
        title="🌸 ¡Propuesta de matrimonio! 💕",
        description=f"¡{interaction.user.mention} le propone matrimonio a {usuario.mention}! UwU\n*¿Qué dices, {usuario.mention}?* 💖✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    view = MarriageView()
    await interaction.response.send_message(embed=embed, view=view)

@bot.tree.command(name="ship", description="🌸 Calcular compatibilidad entre dos usuarios UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def ship_command(interaction: discord.Interaction, usuario1: discord.User, usuario2: discord.User):
    """Calculate ship compatibility"""
    compatibility = bot.affection_manager.calculate_ship_compatibility(usuario1.id, usuario2.id)
    
    # Generate ship name
    name1 = usuario1.display_name[:len(usuario1.display_name)//2]
    name2 = usuario2.display_name[len(usuario2.display_name)//2:]
    ship_name = name1 + name2
    
    # Compatibility messages
    if compatibility >= 90:
        message = "¡Son perfectos el uno para el otro! 💕✨"
        emoji = "💖"
    elif compatibility >= 70:
        message = "¡Muy buena compatibilidad! UwU 💕"
        emoji = "💝"
    elif compatibility >= 50:
        message = "¡Podrían funcionar juntos! >w< 💖"
        emoji = "💘"
    elif compatibility >= 30:
        message = "¡Tal vez como amigos! UwU"
        emoji = "💛"
    else:
        message = "¡Mejor como amigos! >.<"
        emoji = "💙"
    
    embed = discord.Embed(
        title=f"💕 ✨ Ship Kawaii: {ship_name} ✨ 💕",
        description=(
            f"╔════════ 💖 ════════╗\n"
            f"  **{usuario1.mention}**\n"
            f"       💘 + 💘\n"
            f"  **{usuario2.mention}**\n"
            f"╚════════ 💖 ════════╝"
        ),
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    # Create visual heart bar with better visuals
    filled = int(compatibility / 10)
    hearts = "💖" * filled + "🤍" * (10 - filled)
    
    embed.add_field(
        name="💕 Medidor de Amor",
        value=f"{hearts}\n**{compatibility}%** de compatibilidad",
        inline=False
    )
    
    embed.add_field(
        name=f"{emoji} Resultado",
        value=f"```{message}```",
        inline=False
    )
    
    # Add romantic footer
    embed.set_footer(text=f"💖 Ship calculado con amor por Sakura IA • ¡El amor está en el aire! 💖")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="divorce", description="🌸 Divorciarse tristemente de alguien >.<")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def divorce_command(interaction: discord.Interaction, usuario: discord.User):
    """Divorce from someone with GIF"""
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    if not bot.affection_manager.is_married(interaction.user.id, usuario.id, guild_id):
        embed = discord.Embed(
            title="🌸 ¡No están casados! UwU",
            description=f"¡{interaction.user.mention} y {usuario.mention} no están casados! >w<",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    success = bot.affection_manager.remove_marriage(interaction.user.id, usuario.id, guild_id)
    
    if success:
        # Get divorce GIF
        gif_url = await bot.gif_provider.get_gif("cry")
        
        embed = discord.Embed(
            title="🌸 Divorcio completado 💔",
            description=f"¡{interaction.user.mention} y {usuario.mention} se han divorciado! >.<\n*Pero seguirán siendo amigos kawaii* UwU 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        if gif_url:
            embed.set_image(url=gif_url)
    else:
        embed = discord.Embed(
            title="🌸 ¡Oopsie! UwU",
            description="¡Algo salió mal con el divorcio! >.<",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="marriages", description="🌸 Ver lista de matrimonios en el servidor UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def marriages_list_command(interaction: discord.Interaction):
    """Show all marriages in the server"""
    marriages = []
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    
    # Only show marriages for servers, not DMs
    if interaction.guild:
        # Get all members and check for marriages
        for member in interaction.guild.members:
            if member.bot:
                continue
            
            for other_member in interaction.guild.members:
                if (other_member.bot or 
                    other_member.id <= member.id):  # Avoid duplicates
                    continue
                
                if bot.affection_manager.is_married(member.id, other_member.id, guild_id):
                    marriages.append((member, other_member))
    
    embed = discord.Embed(
        title="🌸 Lista de matrimonios kawaii",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if not interaction.guild:
        embed.description = "¡Este comando solo funciona en servidores! >w<\n¡Ve a un servidor y usa `/marriages` allí! 💖"
    elif not marriages:
        embed.description = "¡No hay matrimonios en este servidor todavía! >w<\n¡Usa `/marry` para proponer matrimonio! 💖"
    else:
        marriage_text = ""
        for i, (user1, user2) in enumerate(marriages[:10], 1):
            marriage_text += f"{i}. 💕 {user1.mention} ♡ {user2.mention}\n"
        
        embed.description = f"¡{len(marriages)} parejas felices en este servidor! UwU\n\n{marriage_text}"
        
        if len(marriages) > 10:
            embed.set_footer(text=f"Mostrando 10 de {len(marriages)} matrimonios")
    
    # Add cute GIF
    gif_url = await bot.gif_provider.get_gif("love")
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

# 🌸 Individual Act Commands 🌸
@bot.tree.command(name="cry", description="🌸 Llorar de forma kawaii UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def cry_command(interaction: discord.Interaction):
    """Cry action"""
    gif_url = await bot.gif_provider.get_gif("cry")
    
    embed = discord.Embed(
        title="🌸 ¡Llanto kawaii! 😢",
        description=f"¡{interaction.user.mention} está llorando! *sniff* UwU... 💔",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="dance", description="🌸 Bailar con alegría UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def dance_command(interaction: discord.Interaction):
    """Dance action"""
    gif_url = await bot.gif_provider.get_gif("dance")
    
    embed = discord.Embed(
        title="🌸 ¡Baile kawaii! 💃",
        description=f"¡{interaction.user.mention} está bailando felizmente! ✨\n*¡Qué energía tan linda!* >w< 💖",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="sleep", description="🌸 Dormirse cómodamente UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def sleep_command(interaction: discord.Interaction):
    """Sleep action"""
    gif_url = await bot.gif_provider.get_gif("sleep")
    
    embed = discord.Embed(
        title="🌸 ¡Hora de dormir! 😴",
        description=f"¡{interaction.user.mention} se va a dormir! *Dulces sueños~* UwU 💤",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="laugh", description="🌸 Reírse con alegría UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def laugh_command(interaction: discord.Interaction):
    """Laugh action"""
    gif_url = await bot.gif_provider.get_gif("laugh")
    
    embed = discord.Embed(
        title="🌸 ¡Risas kawaii! 😂",
        description=f"¡{interaction.user.mention} se está riendo! *¡Qué risa tan contagiosa!* UwU ✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="blush", description="🌸 Sonrojarse de forma tierna UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def blush_command(interaction: discord.Interaction):
    """Blush action"""
    try:
        gif_url = await bot.gif_provider.get_gif("blush")
        
        embed = discord.Embed(
            title="🌸 ¡Sonrojo kawaii! 😊",
            description=f"¡{interaction.user.mention} se está sonrojando! *¡Qué tierno!* >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        if gif_url:
            embed.set_image(url=gif_url)
        
        await interaction.response.send_message(embed=embed)
    except Exception as e:
        logger.error(f"Error in blush command: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message("Hubo un error con el comando blush. Intenta de nuevo.", ephemeral=True)
        else:
            await interaction.followup.send("Hubo un error con el comando blush. Intenta de nuevo.", ephemeral=True)

@bot.tree.command(name="wave", description="🌸 Saludar con la manita UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def wave_command(interaction: discord.Interaction):
    """Wave action"""
    gif_url = await bot.gif_provider.get_gif("wave")
    
    embed = discord.Embed(
        title="🌸 ¡Saludo kawaii! 👋",
        description=f"¡{interaction.user.mention} está saludando! *¡Hola hola!* UwU ✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

# 🌸 Individual Interact Commands 🌸
@bot.tree.command(name="huguser", description="🌸 Dar un abrazo tierno a alguien UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def huguser_command(interaction: discord.Interaction, usuario: discord.User):
    """Give a hug with affection tracking and GIF"""

    
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="🌸 ¡Auto-abrazo! UwU",
            description=f"¡{interaction.user.mention} se abraza a sí misma! *¡Amor propio!* >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Update affection stats (use special ID for DMs)
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    bot.affection_manager.update_affection(interaction.user.id, guild_id, "hug_given")
    bot.affection_manager.update_affection(usuario.id, guild_id, "hug_received")
    
    gif_url = await bot.gif_provider.get_gif("hug")
    
    embed = discord.Embed(
        title="🌸 ¡Abrazo kawaii! 🤗",
        description=f"¡{interaction.user.mention} le da un abrazo súper tierno a {usuario.mention}! UwU 💖",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    # Add reaction buttons
    class HugView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)  # 5 minutes instead of 1 minute
        
        async def on_timeout(self):
            """Called when the view times out"""
            # Update embed to show timeout
            embed = discord.Embed(
                title="🌸 ¡Abrazo expirado! >.<",
                description="*susurra tímidamente* Este abrazo ya expiró~ pero puedes enviar uno nuevo uwu ✨",
                color=0xC0C0C0
            )
            embed.set_footer(text="💖 Usa /huguser de nuevo para enviar un abrazo fresco")
            try:
                await self.message.edit(embed=embed, view=None)
            except:
                pass  # Message might be deleted
        
        @discord.ui.button(emoji="💗", style=discord.ButtonStyle.success, label="Devolver")
        async def return_hug(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona abrazada puede devolver el abrazo! UwU", ephemeral=True)
                return
            
            # Check if interaction is still valid
            try:
                if interaction_inner.response.is_done():
                    await interaction_inner.followup.send("*susurra* Esta interacción ya expiró~ Usa /huguser para un abrazo nuevo uwu", ephemeral=True)
                    return
            except:
                pass
            
            # Update affection stats for return hug
            guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
            bot.affection_manager.update_affection(usuario.id, guild_id, "hug_given")
            bot.affection_manager.update_affection(interaction.user.id, guild_id, "hug_received")
            
            # Get updated stats for both users
            stats_returner = bot.affection_manager.get_stats(usuario.id, guild_id)
            stats_receiver = bot.affection_manager.get_stats(interaction.user.id, guild_id)
            
            # Get new gif for return action
            return_gif_url = await bot.gif_provider.get_gif("hug")
            
            embed = discord.Embed(
                title="🌸 ¡Abrazo devuelto! 💗",
                description=f"{usuario.mention} le regresó el abrazo a {interaction.user.mention}! *¡Qué tierno intercambio de cariño!* UwU 💖 **{stats_returner['hugs_given']} abrazos dados**",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if return_gif_url:
                embed.set_image(url=return_gif_url)
            await interaction_inner.response.edit_message(embed=embed, view=None)
        
        @discord.ui.button(emoji="🥀", style=discord.ButtonStyle.secondary, label="Rechazar")
        async def reject_hug(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona abrazada puede rechazar! UwU", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🌸 Abrazo rechazado 🥀",
                description=f"{usuario.mention} rechazó el abrazo de {interaction.user.mention}. *¡Respetamos los límites personales!* UwU 💖",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction_inner.response.edit_message(embed=embed, view=None)
    
    view = HugView()
    await interaction.response.send_message(embed=embed, view=view)
    # Store the message reference in the view for timeout handling
    message = await interaction.original_response()
    view.message = message

@bot.tree.command(name="kissuser", description="🌸 Dar un besito tierno a alguien UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def kissuser_command(interaction: discord.Interaction, usuario: discord.User):
    """Give a kiss with affection tracking and GIF"""

    
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="🌸 ¡Auto-besito! UwU",
            description=f"¡{interaction.user.mention} se da un besito a sí misma! *¡Amor propio!* >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Update affection stats (use special ID for DMs)
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    bot.affection_manager.update_affection(interaction.user.id, guild_id, "kiss_given")
    bot.affection_manager.update_affection(usuario.id, guild_id, "kiss_received")
    
    gif_url = await bot.gif_provider.get_gif("kiss")
    
    embed = discord.Embed(
        title="🌸 ¡Besito kawaii! 😘",
        description=f"¡{interaction.user.mention} le manda un besito volador a {usuario.mention}! Muah~ 💖✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    # Add reaction buttons
    class KissView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)  # 5 minutes instead of 1 minute
        
        async def on_timeout(self):
            """Called when the view times out"""
            embed = discord.Embed(
                title="🌸 ¡Besito expirado! >.<",
                description="*se sonroja* Este besito ya expiró~ pero puedes enviar uno nuevo uwu ✨",
                color=0xC0C0C0
            )
            embed.set_footer(text="💖 Usa /kissuser de nuevo para enviar un besito fresco")
            try:
                await self.message.edit(embed=embed, view=None)
            except:
                pass
        
        @discord.ui.button(emoji="❤️", style=discord.ButtonStyle.success, label="Devolver")
        async def return_kiss(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona besada puede devolver el besito! UwU", ephemeral=True)
                return
            
            # Check if interaction is still valid
            try:
                if interaction_inner.response.is_done():
                    await interaction_inner.followup.send("*susurra* Esta interacción ya expiró~ Usa /kissuser para un besito nuevo uwu", ephemeral=True)
                    return
            except:
                pass
            
            # Update affection stats for return kiss
            guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
            bot.affection_manager.update_affection(usuario.id, guild_id, "kiss_given")
            bot.affection_manager.update_affection(interaction.user.id, guild_id, "kiss_received")
            
            # Get updated stats for both users
            stats_returner = bot.affection_manager.get_stats(usuario.id, guild_id)
            stats_receiver = bot.affection_manager.get_stats(interaction.user.id, guild_id)
            
            # Get new gif for return action
            return_gif_url = await bot.gif_provider.get_gif("kiss")
            
            embed = discord.Embed(
                title="🌸 ¡Besito devuelto! ❤️",
                description=f"{usuario.mention} le regresó el beso a {interaction.user.mention}! *¡Qué romántico intercambio!* UwU Muah~ ✨ **{stats_returner['kisses_given']} besos dados** 💋",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if return_gif_url:
                embed.set_image(url=return_gif_url)
            await interaction_inner.response.edit_message(embed=embed, view=None)
        
        @discord.ui.button(emoji="💔", style=discord.ButtonStyle.secondary, label="Rechazar")
        async def reject_kiss(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona besada puede rechazar! UwU", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🌸 Besito rechazado 💔",
                description=f"{usuario.mention} rechazó el beso de {interaction.user.mention}. *¡Respetamos los límites personales!* UwU 💖",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction_inner.response.edit_message(embed=embed, view=None)
    
    view = KissView()
    await interaction.response.send_message(embed=embed, view=view)
    # Store the message reference in the view for timeout handling
    message = await interaction.original_response()
    view.message = message

@bot.tree.command(name="pat", description="🌸 Dar palmaditas cariñosas UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def pat_command(interaction: discord.Interaction, usuario: discord.User):
    """Pat someone with buttons to return the pat"""

    
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="🌸 ¡Auto-caricia! UwU",
            description=f"¡{interaction.user.mention} se da palmaditas a sí misma! *¡Amor propio!* >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Update affection stats (use special ID for DMs)
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    bot.affection_manager.update_affection(interaction.user.id, guild_id, "pat_given")
    bot.affection_manager.update_affection(usuario.id, guild_id, "pat_received")
    
    gif_url = await bot.gif_provider.get_gif("pat")
    
    embed = discord.Embed(
        title="🌸 ¡Palmaditas kawaii! ✋",
        description=f"¡{interaction.user.mention} le da palmaditas cariñosas a {usuario.mention}! *pat pat* UwU 💖",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    # Add reaction buttons
    class PatView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=300)  # 5 minutes instead of 1 minute
        
        async def on_timeout(self):
            """Called when the view times out"""
            embed = discord.Embed(
                title="🌸 ¡Palmaditas expiradas! >.<",
                description="*esconde las manitas* Estas palmaditas ya expiraron~ pero puedes dar nuevas uwu ✨",
                color=0xC0C0C0
            )
            embed.set_footer(text="💖 Usa /pat de nuevo para dar palmaditas frescas")
            try:
                await self.message.edit(embed=embed, view=None)
            except:
                pass
        
        @discord.ui.button(emoji="🤗", style=discord.ButtonStyle.success, label="Devolver")
        async def return_pat(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona acariciada puede devolver las palmaditas! UwU", ephemeral=True)
                return
            
            # Check if interaction is still valid
            try:
                if interaction_inner.response.is_done():
                    await interaction_inner.followup.send("*susurra* Esta interacción ya expiró~ Usa /pat para palmaditas nuevas uwu", ephemeral=True)
                    return
            except:
                pass
            
            # Update affection stats for return pat
            guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
            bot.affection_manager.update_affection(usuario.id, guild_id, "pat_given")
            bot.affection_manager.update_affection(interaction.user.id, guild_id, "pat_received")
            
            # Get updated stats for both users
            stats_returner = bot.affection_manager.get_stats(usuario.id, guild_id)
            stats_receiver = bot.affection_manager.get_stats(interaction.user.id, guild_id)
            
            # Get new gif for return action
            return_gif_url = await bot.gif_provider.get_gif("pat")
            
            embed = discord.Embed(
                title="🌸 ¡Caricia devuelta! 🤗",
                description=f"{usuario.mention} le regresó la caricia a {interaction.user.mention}! *¡Qué tierno intercambio de cariños!* UwU 💖 **{stats_returner.get('pats_given', 0)} caricias dadas**",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if return_gif_url:
                embed.set_image(url=return_gif_url)
            
            await interaction_inner.response.edit_message(embed=embed, view=None)
        
        @discord.ui.button(emoji="😅", style=discord.ButtonStyle.secondary, label="Rechazar")
        async def reject_pat(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona acariciada puede rechazar! UwU", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🌸 Caricia rechazada 😅",
                description=f"{usuario.mention} rechazó las palmaditas de {interaction.user.mention}. *¡Respetamos los límites personales!* UwU 💖",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction_inner.response.edit_message(embed=embed, view=None)
    
    view = PatView()
    await interaction.response.send_message(embed=embed, view=view)
    # Store the message reference in the view for timeout handling
    message = await interaction.original_response()
    view.message = message

@bot.tree.command(name="poke", description="🌸 Hacer cosquillitas UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def poke_command(interaction: discord.Interaction, usuario: discord.User):
    """Poke someone"""
    gif_url = await bot.gif_provider.get_gif("poke")
    
    embed = discord.Embed(
        title="🌸 ¡Cosquillitas kawaii! 👆",
        description=f"¡{interaction.user.mention} le hace cosquillitas a {usuario.mention}! *poke poke* >w< ✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="cuddle", description="🌸 Acurrucarse tiernamente con alguien UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def cuddle_command(interaction: discord.Interaction, usuario: discord.User):
    """Cuddle with someone with buttons to return the cuddle"""
    if usuario.id == interaction.user.id:
        embed = discord.Embed(
            title="🌸 ¡Auto-abrazo acurrucado! UwU",
            description=f"¡{interaction.user.mention} se acurruca solita! *¡Necesitas compañía!* >w< 💖",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Update affection stats (use special ID for DMs)
    guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
    bot.affection_manager.update_affection(interaction.user.id, guild_id, "cuddle_given")
    bot.affection_manager.update_affection(usuario.id, guild_id, "cuddle_received")
    
    gif_url = await bot.gif_provider.get_gif("cuddle")
    
    embed = discord.Embed(
        title="🌸 ¡Abrazo acurrucado kawaii! 🤗💕",
        description=f"¡{interaction.user.mention} se acurruca tiernamente con {usuario.mention}! *¡Qué calientito y cómodo!* UwU 💖✨",
        color=COLORES_KAWAII["ROSA_KAWAII"]
    )
    
    if gif_url:
        embed.set_image(url=gif_url)
    
    # Add reaction buttons
    class CuddleView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)
        
        @discord.ui.button(emoji="🥰", style=discord.ButtonStyle.success, label="Acurrucarse juntos")
        async def return_cuddle(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona acurrucada puede devolverlo! UwU", ephemeral=True)
                return
            
            # Update affection stats for return cuddle
            guild_id = interaction.guild.id if interaction.guild else 0  # 0 for DMs
            bot.affection_manager.update_affection(usuario.id, guild_id, "cuddle_given")
            bot.affection_manager.update_affection(interaction.user.id, guild_id, "cuddle_received")
            
            # Get updated stats
            stats_returner = bot.affection_manager.get_stats(usuario.id, guild_id)
            
            # Get new gif for return action
            return_gif_url = await bot.gif_provider.get_gif("cuddle")
            
            embed = discord.Embed(
                title="🌸 ¡Acurrucados juntos! 🥰💕",
                description=f"{usuario.mention} se acurrucó de vuelta con {interaction.user.mention}! *¡Ahora están súper cómodos juntos!* UwU 💖 **{stats_returner.get('cuddles_given', 0)} acurrucadas dadas**",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            
            if return_gif_url:
                embed.set_image(url=return_gif_url)
            
            await interaction_inner.response.edit_message(embed=embed, view=None)
        
        @discord.ui.button(emoji="😴", style=discord.ButtonStyle.secondary, label="Prefiero dormir solo")
        async def reject_cuddle(self, interaction_inner: discord.Interaction, button: discord.ui.Button):
            if interaction_inner.user.id != usuario.id:
                await interaction_inner.response.send_message("¡Solo la persona puede rechazar! UwU", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🌸 Prefiere espacio personal 😴",
                description=f"{usuario.mention} prefiere dormir solito por ahora. *¡Respetamos los espacios personales!* UwU 💤",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await interaction_inner.response.edit_message(embed=embed, view=None)
    
    view = CuddleView()
    await interaction.response.send_message(embed=embed, view=view)

# 🌸 Enhanced AI Chat with Kawaii Personality 🌸
# Comando chat removido - ahora usar /sakura con modo chat

# ====================================
# PERSONALITY MANAGEMENT COMMANDS
# ====================================







# Enhanced Image Generation Command
@bot.tree.command(name="generar_imagen", description="🌸 Genera imágenes usando IA - Sakura, Llama y HuggingFace")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    prompt="Descripción de la imagen que quieres generar",
    servicio="Servicio de generación de imágenes"
)
@app_commands.choices(servicio=[
    app_commands.Choice(name="🎨 SDXL Base (Recomendado)", value="huggingface_sdxl"),
    app_commands.Choice(name="🌸 Sakura Style SDXL (Auto)", value="sakura_auto"),
    app_commands.Choice(name="🤗 Stable Diffusion V1.5", value="huggingface_sd"),
    app_commands.Choice(name="🎭 OpenJourney V4", value="openjourney"),
    app_commands.Choice(name="⚡ Llama Vision (OpenRouter)", value="llama_vision")
])
async def generate_image_command(interaction: discord.Interaction, prompt: str, servicio: str = "huggingface_sdxl"):
    """Generate images using Hugging Face models"""
    await interaction.response.defer()
    
    if not bot.ai_provider.huggingface_api_key:
        embed = discord.Embed(
            title="❌ API Key requerida",
            description="Se necesita configurar HUGGINGFACE_API_KEY para usar esta función",
            color=0xff6b9d
        )
        await interaction.followup.send(embed=embed)
        return
    
    try:
        # Add Sakura style prompts if auto mode is selected
        if servicio == "sakura_auto":
            prompt = f"kawaii anime girl, cherry blossoms, pink and white colors, cute, {prompt}, masterpiece, high quality"
            modelo = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL for Sakura style
        elif servicio == "huggingface_sdxl":
            modelo = "stabilityai/stable-diffusion-xl-base-1.0"
        elif servicio == "huggingface_sd":
            modelo = "runwayml/stable-diffusion-v1-5"
        elif servicio == "openjourney":
            modelo = "prompthero/openjourney-v4"
        else:
            modelo = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL as default
        
        # Generate image using Hugging Face with fallback models
        image_bytes = await bot.ai_provider.generate_huggingface_image_with_fallback(prompt, modelo)
        
        if image_bytes:
            # Create file from bytes
            file = discord.File(BytesIO(image_bytes), filename="sakura_generated.png")
            
            embed = discord.Embed(
                title="🎨 Imagen generada por Sakura IA",
                description=f"**Prompt:** {prompt}\n**Modelo:** {modelo}",
                color=0xff6b9d
            )
            embed.set_image(url="attachment://sakura_generated.png")
            embed.set_footer(text="Generado con Hugging Face & Sakura IA 🌸")
            
            await interaction.followup.send(embed=embed, file=file)
        else:
            embed = discord.Embed(
                title="❌ Error generando imagen",
                description="No se pudo generar la imagen. Intenta de nuevo más tarde.",
                color=0xff6b9d
            )
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        embed = discord.Embed(
            title="❌ Error",
            description="Ocurrió un error al generar la imagen",
            color=0xff6b9d
        )
        await interaction.followup.send(embed=embed)



# Contextual Help Bubble with Cute Mascot
class HelpView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=300)
        
    @discord.ui.select(
        placeholder="🌸 ¿En qué puedo ayudarte? Selecciona una categoría...",
        options=[
            discord.SelectOption(
                label="Comandos de IA",
                value="ai_commands",
                description="Aprende sobre los proveedores de IA disponibles",
                emoji="🤖"
            ),
            discord.SelectOption(
                label="Generación de Imágenes",
                value="image_generation", 
                description="Cómo crear imágenes con Hugging Face",
                emoji="🎨"
            ),
            discord.SelectOption(
                label="Búsqueda y Multimedia",
                value="search_commands",
                description="Comandos de búsqueda de videos, imágenes y música",
                emoji="🔍"
            ),
            discord.SelectOption(
                label="Interacciones Kawaii",
                value="kawaii_commands",
                description="Comandos de abrazos, besos y interacciones",
                emoji="💖"
            ),
            discord.SelectOption(
                label="Moderación",
                value="moderation_commands",
                description="Comandos de moderación del servidor",
                emoji="🛡️"
            ),
            discord.SelectOption(
                label="Configuración del Bot",
                value="bot_config",
                description="Personalidades y configuraciones",
                emoji="⚙️"
            )
        ]
    )
    async def help_select(self, interaction: discord.Interaction, select: discord.ui.Select):
        category = select.values[0]
        
        help_data = {
            "ai_commands": {
                "title": "🤖 Comandos de IA - Sakura IA",
                "description": "Sakura IA puede usar múltiples proveedores de IA para responder tus preguntas:",
                "fields": [
                    ("🌟 Comando Principal", "`/ai [proveedor] [pregunta]`\n**Proveedores:** openai, anthropic, xai, gemini, vertex, nvidia, huggingface"),
                    ("💬 Chat Casual", "`/chat [mensaje]` - Habla directamente con Sakura IA"),
                    ("🎯 Mejores Proveedores", "• **OpenRouter**: Más confiable\n• **Hugging Face**: Modelos open-source\n• **NVIDIA**: Modelos avanzados")
                ]
            },
            "image_generation": {
                "title": "🎨 Generación de Imágenes",
                "description": "Crea imágenes increíbles usando modelos de Hugging Face:",
                "fields": [
                    ("🖼️ Comando", "`/generar_imagen [prompt] [modelo]`"),
                    ("🎭 Modelos Disponibles", "• `stabilityai/stable-diffusion-2-1`\n• `runwayml/stable-diffusion-v1-5`"),
                    ("💡 Tips", "• Describe detalladamente lo que quieres\n• Usa inglés para mejores resultados\n• Especifica estilo artístico")
                ]
            },
            "search_commands": {
                "title": "🔍 Búsqueda y Multimedia",
                "description": "Encuentra contenido en internet de manera avanzada:",
                "fields": [
                    ("🎵 Música", "`/search [canción]` - Buscar música\n`/youtube [búsqueda]` - Videos (hasta 100)"),
                    ("🖼️ Imágenes", "`/images [búsqueda]` - Imágenes web (hasta 100)"),
                    ("🌐 Web", "`/web [búsqueda]` - Búsqueda web general")
                ]
            },
            "kawaii_commands": {
                "title": "💖 Interacciones Kawaii",
                "description": "Expresa tus sentimientos de manera adorable:",
                "fields": [
                    ("🤗 Afecto", "`/hug [usuario]` - Abrazo\n`/kiss [usuario]` - Beso"),
                    ("✋ Cariño", "`/pat [usuario]` - Caricias\n`/poke [usuario]` - Cosquillas"),
                    ("📊 Estadísticas", "`/affection [usuario]` - Ver nivel de cariño")
                ]
            },
            "moderation_commands": {
                "title": "🛡️ Moderación",
                "description": "Herramientas para mantener tu servidor seguro:",
                "fields": [
                    ("⚡ Acciones", "`/ban`, `/kick`, `/timeout` - Sanciones\n`/warn` - Advertencias"),
                    ("🗑️ Limpieza", "`/clear [cantidad]` - Borrar mensajes"),
                    ("📊 Información", "`/userinfo`, `/serverinfo` - Información detallada")
                ]
            },
            "bot_config": {
                "title": "⚙️ Configuración del Bot",
                "description": "Personaliza la experiencia con Sakura IA:",
                "fields": [
                    ("🎭 Personalidades", "`/setpersonality` - Cambiar estilo\n`/listpersonalities` - Ver opciones"),
                    ("👋 Interacción", "`/hola` - Saludo personalizado"),
                    ("💖 Tipos", "🌸 Waifu, 🫖 Maid, 🌈 Femboy, 💼 Normal")
                ]
            }
        }
        
        data = help_data[category]
        embed = discord.Embed(
            title=data["title"],
            description=data["description"],
            color=0xff6b9d
        )
        
        for field_name, field_value in data["fields"]:
            embed.add_field(name=field_name, value=field_value, inline=False)
        
        # Add cute mascot
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/1234567890.png")  # Placeholder
        embed.set_footer(text="Sakura IA - Tu asistente kawaii 🌸 | Usa los menús para navegar")
        
        await interaction.response.edit_message(embed=embed, view=self)



# Run the bot
# Ensemble LLM Commands
# Comando ensemble removido - ahora usar /sakura con modo ensemble



# Music Commands - Integrated with Wavelink
@bot.tree.command(name="play", description="🎵 Reproduce música desde YouTube, SoundCloud, etc.")
@app_commands.describe(query="Canción, artista, URL o búsqueda")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
async def play_music_command(interaction: discord.Interaction, query: str):
    """Comando para reproducir música con Wavelink"""
    await interaction.response.defer(thinking=True)
    
    try:
        if not interaction.user.voice:
            embed = discord.Embed(
                title="❌ ¡No estás en un canal de voz!",
                description="¡Baka~! Necesitas estar en un canal de voz para que pueda reproducir música (>.<)",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            return
        
        channel = interaction.user.voice.channel
        
        if not interaction.guild.voice_client:
            player = await channel.connect(cls=wavelink.Player)
            # Optimizar calidad de audio
            player.autoplay = wavelink.AutoPlayMode.enabled
        else:
            player = interaction.guild.voice_client
        
        try:
            # Enhanced search with multiple sources for better results
            search = None
            
            # Use only SoundCloud to avoid YouTube cipher issues
            search = None
            search_queries = [
                f"scsearch:{query}",  # SoundCloud only
                query  # Generic search fallback
            ]
            
            for search_query in search_queries:
                try:
                    search = await wavelink.Playable.search(search_query)
                    if search:
                        logger.info(f"Found track using: {search_query}")
                        break
                except Exception as e:
                    logger.warning(f"Search failed for {search_query}: {e}")
                    continue
            if not search:
                embed = discord.Embed(
                    title="❌ ¡No encontré nada!",
                    description=f"¡Gomen nasai~! No encontré resultados para: `{query}` (T_T)",
                    color=COLORES_KAWAII["ERROR_KAWAII"]
                )
                await interaction.followup.send(embed=embed)
                return
            
            track = search[0]
            
            if not player.playing and not player.paused:
                await player.play(track)
                # Skip audio filters to avoid compatibility issues
                logger.info(f"Playing: {track.title}")
            else:
                # Add to queue with better handling
                if hasattr(player, 'queue'):
                    await player.queue.put_wait(track)
                else:
                    # Create queue if it doesn't exist
                    player.queue = wavelink.Queue()
                    await player.queue.put_wait(track)
            
            embed = discord.Embed(
                title="🎵 ✨ ¡Música Agregada! ✨ UwU",
                description=(
                    f"╔═══════════ ♪ ═══════════╗\n"
                    f"  {'🎸 **¡Reproduciendo Ahora!**' if not player.playing else '📋 **¡Agregado a la Cola!**'}\n"
                    f"╚═══════════ ♪ ═══════════╝\n\n"
                    f"🎶 **{track.title}**"
                ),
                color=COLORES_KAWAII["ROSA_PASTEL"]
            )
            
            if hasattr(track, 'author') and track.author:
                embed.add_field(name="🎤 Artista", value=f"```{track.author}```", inline=True)
            if hasattr(track, 'length') and track.length:
                embed.add_field(name="⏱️ Duración", value=f"```{track.length//60000}:{(track.length%60000)//1000:02d}```", inline=True)
            if hasattr(player, 'queue') and player.queue:
                embed.add_field(name="📊 Posición", value=f"```#{len(player.queue) + 1}```", inline=True)
            
            embed.set_footer(text=f"♪ ¡Disfruta la música! UwU 💕 • Solicitado por {interaction.user.display_name}")
            await interaction.followup.send(embed=embed)
            
        except Exception as search_error:
            embed = discord.Embed(
                title="💔 ¡Error de Búsqueda!",
                description=f"¡Gomen nasai~! Error al buscar: {str(search_error)[:200]}",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        embed = discord.Embed(
            title="💔 ¡Error Reproduciendo!",
            description=f"¡Gomen nasai~! Ocurrió un error: {str(e)[:200]}",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.followup.send(embed=embed)

@bot.tree.command(name="pause", description="⏸️ Pausa la música actual")
async def pause_music_command(interaction: discord.Interaction):
    """Pausar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.playing:
        embed = discord.Embed(
            title="❌ ¡No hay música!",
            description="¡Baka~! No hay música reproduciéndose (>.<)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(True)
    embed = discord.Embed(
        title="⏸️ ¡Música Pausada!",
        description="¡Hai hai~! Pausé la música para ti (＾◡＾)",
        color=COLORES_KAWAII["ALERTA_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="resume", description="▶️ Reanuda la música pausada")
async def resume_music_command(interaction: discord.Interaction):
    """Reanudar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.paused:
        embed = discord.Embed(
            title="❌ ¡Música no pausada!",
            description="¡Baka~! La música no está pausada (>.<)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(False)
    embed = discord.Embed(
        title="▶️ ¡Música Reanudada!",
        description="¡Yay~! Continúo reproduciendo música para ti! 🎵",
        color=COLORES_KAWAII["EXITO_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="stop", description="⏹️ Detiene la música y desconecta el bot")
async def stop_music_command(interaction: discord.Interaction):
    """Detener música y desconectar"""
    player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ ¡No estoy conectada!",
            description="¡Baka~! No estoy conectada a un canal de voz (>.<)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.disconnect()
    embed = discord.Embed(
        title="⏹️ ¡Música Detenida!",
        description="¡Hai hai~! Me desconecté del canal de voz (＾◡＾)\n¡Arigato por escuchar música conmigo! 💕",
        color=COLORES_KAWAII["CELESTE_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="volume", description="🔊 Ajustar el volumen de la música")
@app_commands.describe(volume="Volumen del 1 al 100")
async def volume_command(interaction: discord.Interaction, volume: int):
    """Ajustar el volumen de la música"""
    if not interaction.guild.voice_client:
        embed = discord.Embed(
            title="❌ ¡No estoy conectada!",
            description="¡Baka~! No estoy en un canal de voz (>.<)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        return await interaction.response.send_message(embed=embed)
    
    if volume < 1 or volume > 100:
        embed = discord.Embed(
            title="❌ ¡Volumen inválido!",
            description="¡Gomen nasai~! El volumen debe estar entre 1 y 100 UwU",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        return await interaction.response.send_message(embed=embed)
    
    player = interaction.guild.voice_client
    await player.set_volume(volume)
    
    embed = discord.Embed(
        title="🔊 ✨ ¡Volumen Ajustado! ✨",
        description=(
            f"╔═══════════════════════╗\n"
            f"  🎚️ **Volumen: {volume}%**\n"
            f"  {'🔇' if volume == 0 else '🔈' if volume < 30 else '🔉' if volume < 70 else '🔊'} "
            f"{'¡Silenciado!' if volume == 0 else '¡Suave!' if volume < 30 else '¡Moderado!' if volume < 70 else '¡Alto!'}\n"
            f"╚═══════════════════════╝\n\n"
            f"¡Hai hai~! Volumen configurado perfectamente ♪(´▽｀)"
        ),
        color=COLORES_KAWAII["CELESTE_KAWAII"]
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="queue", description="📋 Ver las canciones en cola")
async def queue_music_command(interaction: discord.Interaction):
    """Ver cola de música"""
    player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ ¡No estoy conectada!",
            description="¡Baka~! No estoy conectada a un canal de voz (>.<)",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    if not hasattr(player, 'queue') or not player.queue:
        embed = discord.Embed(
            title="📋 ¡Cola Vacía! (´･ω･`)",
            description=(
                "╔════════════════════════╗\n"
                "  🎵 No hay canciones en cola\n"
                "  📝 Usa `/play` para agregar\n"
                "╚════════════════════════╝\n\n"
                "¡Añade música para empezar la fiesta! UwU 🎉"
            ),
            color=COLORES_KAWAII["CELESTE_KAWAII"]
        )
    else:
        queue_list = []
        for i, track in enumerate(list(player.queue)[:10], 1):
            duration = f"{track.length//60000}:{(track.length%60000)//1000:02d}" if hasattr(track, 'length') else "??:??"
            queue_list.append(f"`{i:02d}` 🎵 **{track.title}**\n     └─ *{track.author}* • `{duration}`")
        
        embed = discord.Embed(
            title="📋 ✨ Cola de Reproducción ✨",
            description=(
                f"🎸 **Canciones en cola: {len(player.queue)}**\n"
                f"{'─' * 30}\n\n"
                + "\n\n".join(queue_list)
            ),
            color=COLORES_KAWAII["ROSA_PASTEL"]
        )
        
        if len(player.queue) > 10:
            embed.set_footer(text=f"📊 Mostrando 10 de {len(player.queue)} canciones • Usa /queue para ver más")
    
    await interaction.response.send_message(embed=embed)

# Advanced Music Commands
@bot.tree.command(name="shuffle", description="🔀 Mezclar la cola de reproducción")
async def shuffle_command(interaction: discord.Interaction):
    """Mezclar cola de reproducción"""
    if not interaction.guild.voice_client:
        embed = discord.Embed(title="❌ No hay música reproduciéndose", color=COLORES_KAWAII["ERROR_KAWAII"])
        await interaction.response.send_message(embed=embed)
        return
    
    player = interaction.guild.voice_client
    if hasattr(player, 'queue') and len(player.queue) > 1:
        player.queue.shuffle()
        embed = discord.Embed(
            title="🔀 Cola mezclada",
            description=f"Se mezclaron {len(player.queue)} canciones en la cola",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
    else:
        embed = discord.Embed(title="❌ No hay suficientes canciones en la cola", color=COLORES_KAWAII["ERROR_KAWAII"])
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="loop", description="🔁 Activar/desactivar repetición")
@app_commands.describe(mode="Modo de repetición: track, queue, off")
async def loop_command(interaction: discord.Interaction, mode: str):
    """Configurar modo de repetición"""
    if not interaction.guild.voice_client:
        embed = discord.Embed(title="❌ No hay música reproduciéndose", color=COLORES_KAWAII["ERROR_KAWAII"])
        await interaction.response.send_message(embed=embed)
        return
    
    player = interaction.guild.voice_client
    
    if mode.lower() == "track":
        player.queue.mode = wavelink.QueueMode.loop
        embed = discord.Embed(title="🔁 Repetir canción actual activado", color=COLORES_KAWAII["EXITO_KAWAII"])
    elif mode.lower() == "queue":
        player.queue.mode = wavelink.QueueMode.loop_all
        embed = discord.Embed(title="🔁 Repetir cola completa activado", color=COLORES_KAWAII["EXITO_KAWAII"])
    elif mode.lower() == "off":
        player.queue.mode = wavelink.QueueMode.normal
        embed = discord.Embed(title="🔁 Repetición desactivada", color=COLORES_KAWAII["EXITO_KAWAII"])
    else:
        embed = discord.Embed(
            title="❌ Modo inválido",
            description="Usa: `track`, `queue` o `off`",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
    
    await interaction.response.send_message(embed=embed)





@bot.tree.command(name="clear", description="🗑️ Limpiar cola de reproducción")
async def clear_queue_command(interaction: discord.Interaction):
    """Limpiar cola de reproducción"""
    if not interaction.guild.voice_client:
        embed = discord.Embed(title="❌ No hay música reproduciéndose", color=COLORES_KAWAII["ERROR_KAWAII"])
        await interaction.response.send_message(embed=embed)
        return
    
    player = interaction.guild.voice_client
    
    if hasattr(player, 'queue') and len(player.queue) > 0:
        queue_size = len(player.queue)
        player.queue.clear()
        embed = discord.Embed(
            title="🗑️ Cola limpiada",
            description=f"Se eliminaron {queue_size} canciones de la cola",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
    else:
        embed = discord.Embed(title="❌ La cola ya está vacía", color=COLORES_KAWAII["ERROR_KAWAII"])
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="remove", description="🗑️ Remover canción de la cola")
@app_commands.describe(position="Posición de la canción en la cola (1, 2, 3...)")
async def remove_track_command(interaction: discord.Interaction, position: int):
    """Remover canción específica de la cola"""
    if not interaction.guild.voice_client:
        embed = discord.Embed(title="❌ No hay música reproduciéndose", color=COLORES_KAWAII["ERROR_KAWAII"])
        await interaction.response.send_message(embed=embed)
        return
    
    player = interaction.guild.voice_client
    
    if not hasattr(player, 'queue') or len(player.queue) == 0:
        embed = discord.Embed(title="❌ La cola está vacía", color=COLORES_KAWAII["ERROR_KAWAII"])
        await interaction.response.send_message(embed=embed)
        return
    
    if position < 1 or position > len(player.queue):
        embed = discord.Embed(
            title="❌ Posición inválida",
            description=f"Elige una posición entre 1 y {len(player.queue)}",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        return
    
    # Convert to 0-based index
    index = position - 1
    removed_track = player.queue[index]
    del player.queue[index]
    
    embed = discord.Embed(
        title="🗑️ Canción removida",
        description=f"**{removed_track.title}** fue removida de la posición {position}",
        color=COLORES_KAWAII["EXITO_KAWAII"]
    )
    
    await interaction.response.send_message(embed=embed)







# ============================================================================
# 🌸 COMANDO DE PRUEBA CLOUDFLARE AI
# ============================================================================



# ============================================================================
# 🛡️ COMANDOS DE AUTOMOD
# ============================================================================

@bot.tree.command(name="crear_filtro", description="🛡️ Crear reglas de AutoMod para filtrar palabras")
@app_commands.describe(
    palabras="Palabras a filtrar (separadas por comas)",
    accion="Acción a tomar cuando se detecte una palabra filtrada"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="Eliminar mensaje", value="delete"),
    app_commands.Choice(name="Solo advertir", value="warn")
])
async def crear_filtro_command(interaction: discord.Interaction, palabras: str, accion: str = "delete"):
    """Crear filtro de palabras de AutoMod"""
    
    # Verificar permisos
    if not interaction.user.guild_permissions.manage_messages:
        embed = discord.Embed(
            title="❌ Sin Permisos",
            description="Necesitas el permiso `Administrar Mensajes` para usar este comando.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        # Procesar lista de palabras
        palabras_lista = [palabra.strip() for palabra in palabras.split(',')]
        palabras_lista = [p for p in palabras_lista if p]
        
        if not palabras_lista:
            raise ValueError("Debes proporcionar al menos una palabra")
        
        # Intentar crear regla de AutoMod nativo primero
        try:
            # Crear regla usando la API de Discord si está disponible
            actions = []
            
            # Configurar acciones según el tipo
            if hasattr(discord, 'AutoModerationAction'):
                if accion == "delete":
                    action = discord.AutoModerationAction(
                        type=discord.AutoModerationActionType.block_message
                    )
                else:  # warn
                    action = discord.AutoModerationAction(
                        type=discord.AutoModerationActionType.send_alert_message
                    )
                
                actions.append(action)
                
                # Crear trigger
                trigger = discord.AutoModerationTrigger(
                    type=discord.AutoModerationTriggerType.keyword,
                    keyword_filter=palabras_lista
                )
                
                # Crear regla
                regla = await interaction.guild.create_automod_rule(
                    name=f"Filtro-{interaction.user.display_name}",
                    event_type=discord.AutoModerationEventType.message_send,
                    trigger=trigger,
                    actions=actions,
                    enabled=True,
                    reason=f"Regla creada por {interaction.user}"
                )
                
                # Guardar en base de datos
                bot.automod_manager.save_native_rule(
                    interaction.guild.id,
                    regla.id,
                    regla.name,
                    "keyword",
                    palabras_lista,
                    interaction.user.id
                )
                
                embed = discord.Embed(
                    title="✅ AutoMod Nativo Creado",
                    description="Se ha creado una regla de AutoMod nativa de Discord.",
                    color=discord.Color.green()
                )
                embed.add_field(name="ID de Regla", value=f"`{regla.id}`", inline=True)
                
            else:
                raise AttributeError("AutoMod nativo no disponible")
                
        except (AttributeError, Exception) as e:
            logger.info(f"AutoMod nativo no disponible o falló: {e}, usando sistema manual")
            
            # Usar sistema de filtrado manual como fallback
            new_words_count = bot.automod_manager.add_filtered_words(interaction.guild.id, palabras_lista)
            bot.automod_manager.set_action_type(interaction.guild.id, accion)
            
            embed = discord.Embed(
                title="✅ Filtro Manual Creado",
                description="Se ha creado un filtro de AutoMod manual (sistema propio).",
                color=discord.Color.orange()
            )
            embed.add_field(name="Palabras Nuevas", value=f"`{new_words_count}`", inline=True)
        
        embed.add_field(
            name="🔍 Palabras Filtradas",
            value=f"```{', '.join(palabras_lista[:10])}{'...' if len(palabras_lista) > 10 else ''}```",
            inline=False
        )
        
        embed.add_field(
            name="⚡ Acción",
            value={
                "delete": "🗑️ Eliminar mensaje",
                "warn": "⚠️ Solo advertir"
            }[accion],
            inline=True
        )
        
        embed.set_footer(text=f"Creado por {interaction.user.display_name}")
        
        await interaction.followup.send(embed=embed)
        
        logger.info(f"Filtro AutoMod creado por {interaction.user} en {interaction.guild.name}")
        
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Ocurrió un error al crear el filtro: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
        logger.error(f"Error creando filtro AutoMod: {e}")

@bot.tree.command(name="automod_config", description="🛡️ Configurar sistema de AutoMod")
@app_commands.describe(
    accion="Acción a configurar",
    canal_logs="Canal para enviar logs de AutoMod (opcional)"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="Ver configuración", value="view"),
    app_commands.Choice(name="Habilitar", value="enable"),
    app_commands.Choice(name="Deshabilitar", value="disable"),
    app_commands.Choice(name="Configurar logs", value="set_logs")
])
async def automod_config_command(interaction: discord.Interaction, accion: str, canal_logs: discord.TextChannel = None):
    """Configurar AutoMod"""
    
    if not interaction.user.guild_permissions.manage_guild:
        embed = discord.Embed(
            title="❌ Sin Permisos",
            description="Necesitas el permiso `Administrar Servidor` para usar este comando.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    config = bot.automod_manager.get_config(interaction.guild.id)
    
    if accion == "view":
        embed = discord.Embed(
            title="🛡️ Configuración de AutoMod",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Estado",
            value="✅ Habilitado" if config['enabled'] else "❌ Deshabilitado",
            inline=True
        )
        
        embed.add_field(
            name="Acción",
            value={"delete": "🗑️ Eliminar", "warn": "⚠️ Advertir"}[config['action_type']],
            inline=True
        )
        
        embed.add_field(
            name="Palabras Filtradas",
            value=f"`{len(config['filtered_words'])}` palabras",
            inline=True
        )
        
        if config['log_channel_id']:
            log_channel = bot.get_channel(config['log_channel_id'])
            embed.add_field(
                name="Canal de Logs",
                value=log_channel.mention if log_channel else "Canal no encontrado",
                inline=False
            )
        
        if config['filtered_words']:
            words_display = ', '.join(config['filtered_words'][:10])
            if len(config['filtered_words']) > 10:
                words_display += f" (+{len(config['filtered_words']) - 10} más)"
            embed.add_field(
                name="Palabras",
                value=f"```{words_display}```",
                inline=False
            )
        
    elif accion == "enable":
        config['enabled'] = True
        bot.automod_manager.save_config(interaction.guild.id, config)
        embed = discord.Embed(
            title="✅ AutoMod Habilitado",
            description="El sistema de AutoMod ha sido habilitado.",
            color=discord.Color.green()
        )
        
    elif accion == "disable":
        config['enabled'] = False
        bot.automod_manager.save_config(interaction.guild.id, config)
        embed = discord.Embed(
            title="❌ AutoMod Deshabilitado",
            description="El sistema de AutoMod ha sido deshabilitado.",
            color=discord.Color.red()
        )
        
    elif accion == "set_logs":
        if not canal_logs:
            embed = discord.Embed(
                title="❌ Error",
                description="Debes especificar un canal para los logs.",
                color=discord.Color.red()
            )
        else:
            bot.automod_manager.set_log_channel(interaction.guild.id, canal_logs.id)
            embed = discord.Embed(
                title="✅ Canal de Logs Configurado",
                description=f"Los logs de AutoMod se enviarán a {canal_logs.mention}.",
                color=discord.Color.green()
            )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="automod_stats", description="📊 Ver estadísticas de AutoMod")
@app_commands.describe(dias="Número de días para las estadísticas (por defecto: 7)")
async def automod_stats_command(interaction: discord.Interaction, dias: int = 7):
    """Ver estadísticas de AutoMod"""
    
    if not interaction.user.guild_permissions.manage_messages:
        embed = discord.Embed(
            title="❌ Sin Permisos",
            description="Necesitas permisos de moderador para ver las estadísticas.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    stats = bot.automod_manager.get_infractions_stats(interaction.guild.id, dias)
    config = bot.automod_manager.get_config(interaction.guild.id)
    
    embed = discord.Embed(
        title=f"📊 Estadísticas de AutoMod ({dias} días)",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(
        name="Total de Infracciones",
        value=f"`{stats['recent_count']}`",
        inline=True
    )
    
    embed.add_field(
        name="Palabras Filtradas",
        value=f"`{len(config['filtered_words'])}`",
        inline=True
    )
    
    embed.add_field(
        name="Estado del Sistema",
        value="✅ Activo" if config['enabled'] else "❌ Inactivo",
        inline=True
    )
    
    if stats['top_user']:
        user = bot.get_user(stats['top_user'][0])
        embed.add_field(
            name="Usuario con más infracciones",
            value=f"{user.mention if user else 'Usuario desconocido'}: `{stats['top_user'][1]}` infracciones",
            inline=False
        )
    
    if stats['top_words']:
        words_text = "\n".join([f"`{word}`: {count}" for word, count in stats['top_words']])
        embed.add_field(
            name="Palabras más detectadas",
            value=words_text,
            inline=False
        )
    
    embed.set_footer(text=f"Solicitado por {interaction.user.display_name}")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="automod_words", description="📝 Gestionar palabras filtradas")
@app_commands.describe(
    accion="Acción a realizar",
    palabras="Palabras a agregar o eliminar (separadas por comas)"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="Listar palabras", value="list"),
    app_commands.Choice(name="Agregar palabras", value="add"),
    app_commands.Choice(name="Eliminar palabras", value="remove"),
    app_commands.Choice(name="Limpiar todas", value="clear")
])
async def automod_words_command(interaction: discord.Interaction, accion: str, palabras: str = None):
    """Gestionar palabras filtradas"""
    
    if not interaction.user.guild_permissions.manage_messages:
        embed = discord.Embed(
            title="❌ Sin Permisos",
            description="Necesitas el permiso `Administrar Mensajes` para usar este comando.",
            color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    config = bot.automod_manager.get_config(interaction.guild.id)
    
    if accion == "list":
        embed = discord.Embed(
            title="📝 Palabras Filtradas",
            color=discord.Color.blue()
        )
        
        if config['filtered_words']:
            # Dividir en chunks para evitar límites de caracteres
            chunks = [config['filtered_words'][i:i+20] for i in range(0, len(config['filtered_words']), 20)]
            
            for i, chunk in enumerate(chunks):
                field_name = f"Palabras ({i+1}/{len(chunks)})" if len(chunks) > 1 else "Palabras"
                embed.add_field(
                    name=field_name,
                    value=f"```{', '.join(chunk)}```",
                    inline=False
                )
            
            embed.add_field(
                name="Total",
                value=f"`{len(config['filtered_words'])}` palabras",
                inline=True
            )
        else:
            embed.description = "No hay palabras filtradas configuradas."
    
    elif accion == "add":
        if not palabras:
            embed = discord.Embed(
                title="❌ Error",
                description="Debes especificar palabras para agregar.",
                color=discord.Color.red()
            )
        else:
            palabras_lista = [p.strip() for p in palabras.split(',') if p.strip()]
            new_count = bot.automod_manager.add_filtered_words(interaction.guild.id, palabras_lista)
            
            embed = discord.Embed(
                title="✅ Palabras Agregadas",
                color=discord.Color.green()
            )
            embed.add_field(
                name="Nuevas palabras",
                value=f"`{new_count}` palabras agregadas",
                inline=True
            )
            embed.add_field(
                name="Total actual",
                value=f"`{len(bot.automod_manager.get_config(interaction.guild.id)['filtered_words'])}` palabras",
                inline=True
            )
    
    elif accion == "remove":
        if not palabras:
            embed = discord.Embed(
                title="❌ Error",
                description="Debes especificar palabras para eliminar.",
                color=discord.Color.red()
            )
        else:
            palabras_lista = [p.strip() for p in palabras.split(',') if p.strip()]
            removed_count = bot.automod_manager.remove_filtered_words(interaction.guild.id, palabras_lista)
            
            embed = discord.Embed(
                title="🗑️ Palabras Eliminadas",
                color=discord.Color.orange()
            )
            embed.add_field(
                name="Palabras eliminadas",
                value=f"`{removed_count}` palabras eliminadas",
                inline=True
            )
            embed.add_field(
                name="Total actual",
                value=f"`{len(bot.automod_manager.get_config(interaction.guild.id)['filtered_words'])}` palabras",
                inline=True
            )
    
    elif accion == "clear":
        bot.automod_manager.clear_filtered_words(interaction.guild.id)
        embed = discord.Embed(
            title="🧹 Palabras Limpiadas",
            description="Se han eliminado todas las palabras filtradas.",
            color=discord.Color.red()
        )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="automod_test", description="🧪 Probar el sistema de AutoMod")
async def automod_test_command(interaction: discord.Interaction):
    """Probar AutoMod"""
    try:
        if not interaction.user.guild_permissions.administrator:
            embed = discord.Embed(
                title="❌ Sin Permisos",
                description="Necesitas permisos de administrador para probar el sistema.",
                color=discord.Color.red()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        config = bot.automod_manager.get_config(interaction.guild.id)
        
        embed = discord.Embed(
            title="🧪 Test del Sistema AutoMod",
            color=discord.Color.orange()
        )
        
        # Estado del sistema
        embed.add_field(
            name="🛡️ Estado",
            value="✅ Activo" if config['enabled'] else "❌ Inactivo",
            inline=True
        )
        
        embed.add_field(
            name="📝 Palabras Filtradas",
            value=f"`{len(config['filtered_words'])}` configuradas",
            inline=True
        )
        
        embed.add_field(
            name="⚡ Acción",
            value={"delete": "🗑️ Eliminar", "warn": "⚠️ Advertir"}[config['action_type']],
            inline=True
        )
        
        # Intents habilitados
        intents_info = []
        if bot.intents.auto_moderation_configuration:
            intents_info.append("✅ AutoMod Configuration")
        if bot.intents.auto_moderation_execution:
            intents_info.append("✅ AutoMod Execution")
        if bot.intents.guild_messages:
            intents_info.append("✅ Guild Messages")
        
        embed.add_field(
            name="🔧 Intents",
            value="\n".join(intents_info) if intents_info else "❌ Intents no disponibles",
            inline=False
        )
        
        # Instrucciones de prueba
        if config['filtered_words']:
            test_words = config['filtered_words'][:3]
            embed.add_field(
                name="🧪 Cómo Probar",
                value=(
                    f"1. Escribe un mensaje con alguna de estas palabras: `{', '.join(test_words)}`\n"
                    f"2. El sistema debería detectarla automáticamente\n"
                    f"3. Se ejecutará la acción: {config['action_type']}\n\n"
                    "**Nota:** Este mensaje no activará los filtros."
                ),
                inline=False
            )
        else:
            embed.add_field(
                name="⚠️ Sin Palabras",
                value="Usa `/crear_filtro` para agregar palabras de prueba.",
                inline=False
            )
        
        await interaction.response.send_message(embed=embed)
    except Exception as e:
        logger.error(f"Error in automod_test command: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message("Hubo un error con el comando AutoMod test. Intenta de nuevo.", ephemeral=True)
        else:
            await interaction.followup.send("Hubo un error con el comando AutoMod test. Intenta de nuevo.", ephemeral=True)

# REMOVED SLASH FORCE_SYNC COMMAND TO PREVENT RATE LIMITS
# This command was causing multiple sync attempts

# 🌸✨ Comando de Descarga Kawaii ✨🌸
@bot.tree.command(name="download", description="📥 Descargar contenido de tu última búsqueda UwU")
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.describe(
    numero="Número del resultado que quieres descargar (1-20)"
)
async def download_command(interaction: discord.Interaction, numero: int = 1):
    """Download content from previous search results"""
    await interaction.response.defer()
    
    try:
        # Verificar si hay búsquedas en cache
        if interaction.user.id not in search_cache:
            embed = discord.Embed(
                title="🌸 ¡Ay no! UwU",
                description="¡No tienes búsquedas recientes para descargar, mi amor! >.<\n\n"
                           "💡 **Usa primero:**\n"
                           "• `/imgsearch` para buscar imágenes\n"
                           "• `/ytsearch` para buscar videos\n"
                           "¡Y luego podrás descargar con `/download`! ♡",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            embed.set_footer(text="¡Haz una búsqueda primero y vuelve! (◕‿◕)♡")
            await interaction.followup.send(embed=embed)
            return
        
        cached_search = search_cache[interaction.user.id]
        results = cached_search["results"]
        search_type = cached_search["type"]
        query = cached_search["query"]
        
        # Verificar si la búsqueda no es muy antigua (30 minutos)
        if datetime.now() - cached_search["timestamp"] > timedelta(minutes=30):
            embed = discord.Embed(
                title="🌸 ¡Búsqueda expirada! UwU",
                description="¡Tu búsqueda es muy antigua, preciosa! >.<\n\n"
                           "Haz una nueva búsqueda y vuelve pronto ♡",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            embed.set_footer(text="Las búsquedas se guardan por 30 minutos (◕‿◕)♡")
            await interaction.followup.send(embed=embed)
            # Limpiar cache expirado
            del search_cache[interaction.user.id]
            return
        
        # Verificar número válido
        if numero < 1 or numero > len(results):
            embed = discord.Embed(
                title="🌸 ¡Número inválido! UwU",
                description=f"¡Ese número no existe, mi amor! >.<\n\n"
                           f"📋 **Tu búsqueda de '{query}' tiene {len(results)} resultados**\n"
                           f"💡 Usa un número entre 1 y {len(results)}",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            embed.set_footer(text="¡Verifica el número y vuelve! (◕‿◕)♡")
            await interaction.followup.send(embed=embed)
            return
        
        # Obtener el resultado específico
        selected_result = results[numero - 1]
        
        # Mostrar mensaje de procesamiento
        embed = discord.Embed(
            title="🌸✨ Sakura IA está descargando... ✨🌸",
            description=f"*preparando tu descarga kawaii* 📥💖\n\n"
                       f"**Búsqueda:** {query}\n"
                       f"**Resultado:** #{numero}\n"
                       f"**Tipo:** {'🖼️ Imagen' if search_type == 'images' else '🎵 Video de YouTube'}",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        processing_message = await interaction.followup.send(embed=embed)
        
        if search_type == "images":
            await download_image(interaction, selected_result, numero, query, processing_message)
        elif search_type == "youtube":
            await download_youtube(interaction, selected_result, numero, query, processing_message)
        
    except Exception as e:
        logger.error(f"Error in download command: {e}")
        embed = discord.Embed(
            title="🌸 ¡Oopsie! UwU",
            description="¡Algo salió mal con la descarga, pero ya lo arreglo! 💔",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        embed.set_footer(text="¡Intenta de nuevo en un momentito! (◕‿◕)♡")
        if 'processing_message' in locals():
            await processing_message.edit(embed=embed)
        else:
            await interaction.followup.send(embed=embed)

async def download_image(interaction, result, numero, query, processing_message):
    """Download and send image"""
    try:
        image_url = result.get('url', '')
        title = result.get('title', 'Sin título')[:100]
        
        if not image_url:
            embed = discord.Embed(
                title="🌸 ¡Error! UwU",
                description="¡Esta imagen no tiene una URL válida, mi amor! >.<",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await processing_message.edit(embed=embed)
            return
        
        # Descargar la imagen
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    
                    # Crear embed de éxito
                    embed = discord.Embed(
                        title="🌸✨ ¡Descarga Completada! ✨🌸",
                        description=f"¡Aquí tienes tu imagen kawaii! ♡\n\n"
                                   f"**📋 Resultado #{numero}**\n"
                                   f"**🔍 Búsqueda:** {query}\n"
                                   f"**🖼️ Título:** {title}",
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    embed.set_footer(text="¡Disfruta tu imagen! (◕‿◕)♡")
                    embed.set_image(url=image_url)
                    
                    # Crear archivo para enviar
                    file_extension = image_url.split('.')[-1] if '.' in image_url else 'jpg'
                    filename = f"sakura_image_{numero}.{file_extension}"
                    
                    file = discord.File(
                        io.BytesIO(image_data), 
                        filename=filename
                    )
                    
                    await processing_message.edit(embed=embed)
                    await interaction.followup.send(
                        f"🌸💖 **Archivo descargado:** `{filename}`",
                        file=file
                    )
                else:
                    embed = discord.Embed(
                        title="🌸 ¡Error de descarga! UwU",
                        description="¡No pude descargar esta imagen, mi amor! >.<\n"
                                   "Quizás el enlace ya no funciona ♡",
                        color=COLORES_KAWAII["ROSA_KAWAII"]
                    )
                    await processing_message.edit(embed=embed)
    
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        embed = discord.Embed(
            title="🌸 ¡Error! UwU", 
            description="¡Hubo un problema descargando la imagen! 💔",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await processing_message.edit(embed=embed)

async def download_youtube(interaction, result, numero, query, processing_message):
    """Download YouTube video/audio"""
    try:
        video_url = result.get('url', '')
        title = result.get('title', 'Sin título')[:100]
        channel = result.get('channel', 'Canal desconocido')
        
        if not video_url:
            embed = discord.Embed(
                title="🌸 ¡Error! UwU",
                description="¡Este video no tiene una URL válida, mi amor! >.<",
                color=COLORES_KAWAII["ROSA_KAWAII"]
            )
            await processing_message.edit(embed=embed)
            return
        
        # Actualizar mensaje con información del video
        embed = discord.Embed(
            title="🌸✨ ¡Descarga Completada! ✨🌸",
            description=f"¡Aquí tienes tu video kawaii! ♡\n\n"
                       f"**📋 Resultado #{numero}**\n"
                       f"**🔍 Búsqueda:** {query}\n"
                       f"**🎵 Título:** {title}\n"
                       f"**📺 Canal:** {channel}\n"
                       f"**🔗 URL:** [Ver en YouTube]({video_url})",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        embed.set_footer(text="¡Nota: Solo puedo mostrar el enlace por limitaciones de Discord! (◕‿◕)♡")
        
        # Obtener thumbnail si está disponible
        thumbnail = result.get('thumbnail')
        if thumbnail:
            embed.set_image(url=thumbnail)
        
        await processing_message.edit(embed=embed)
        
        # Enviar información adicional
        info_text = (
            f"🌸💖 **Video de YouTube encontrado:**\n"
            f"```\n"
            f"Título: {title}\n"
            f"Canal: {channel}\n"
            f"URL: {video_url}\n"
            f"```\n"
            f"💡 **Para descargar:** Copia la URL y usa un descargador de YouTube externo ♡"
        )
        
        await interaction.followup.send(info_text)
        
    except Exception as e:
        logger.error(f"Error processing YouTube result: {e}")
        embed = discord.Embed(
            title="🌸 ¡Error! UwU",
            description="¡Hubo un problema procesando el video! 💔",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        await processing_message.edit(embed=embed)

# Enhanced Resilient Bot Commands
@bot.command(name='health', aliases=['status', 'estado'])
async def health_command(ctx):
    """Show comprehensive bot health status with circuit breaker info"""
    try:
        # Get bot instance and circuit breaker status
        if hasattr(bot, 'sync_circuit_breaker'):
            cb_state = bot.sync_circuit_breaker.state.value
            cb_failures = bot.sync_circuit_breaker.failure_count
            can_sync, reason = bot.sync_circuit_breaker.can_execute()
        else:
            cb_state = "Not initialized"
            cb_failures = 0
            can_sync = False
            reason = "Circuit breaker not available"
        
        embed = discord.Embed(
            title="🌸 Sakura IA - Estado del Sistema Resiliente",
            color=0xFFB6C1,
            timestamp=datetime.now()
        )
        
        # Command status with detailed info
        if getattr(bot, 'commands_synced', False):
            cmd_status = "✅ Slash commands activos"
            status_color = "🟢"
        elif getattr(bot, 'emergency_mode', False):
            cmd_status = "🚫 Modo emergencia - Solo comandos de texto"
            status_color = "🔴"
        else:
            cmd_status = "⏳ Slash commands pendientes"
            status_color = "🟡"
        
        embed.add_field(
            name="📝 Estado de Comandos", 
            value=f"{status_color} {cmd_status}", 
            inline=True
        )
        
        # Circuit breaker detailed status
        cb_emoji = {
            "closed": "🟢",
            "open": "🔴", 
            "half_open": "🟡"
        }.get(cb_state.lower(), "⚪")
        
        embed.add_field(
            name="🛡️ Circuit Breaker",
            value=f"{cb_emoji} {cb_state.title()}\nFallos: {cb_failures}",
            inline=True
        )
        
        # Sync capability
        sync_status = "✅ Disponible" if can_sync else f"❌ Bloqueado\n{reason}"
        embed.add_field(
            name="🔄 Capacidad de Sync",
            value=sync_status,
            inline=True
        )
        
        # Bot performance stats
        uptime = datetime.now() - datetime.fromtimestamp(bot.startup_time) if hasattr(bot, 'startup_time') else "Unknown"
        embed.add_field(
            name="📊 Estadísticas",
            value=f"Latencia: {round(bot.latency * 1000)}ms\n"
                  f"Servidores: {len(bot.guilds)}\n"
                  f"Tiempo activo: {str(uptime).split('.')[0] if uptime != 'Unknown' else 'Unknown'}",
            inline=True
        )
        
        # Rate limiter status
        if hasattr(bot, 'rate_limiter'):
            global_reqs = len(bot.rate_limiter.global_requests)
            blocked_until = bot.rate_limiter.blocked_until
            is_blocked = time.time() < blocked_until
            
            rate_status = f"Requests: {global_reqs}/{bot.rate_limiter.global_limit}\n"
            if is_blocked:
                remaining = blocked_until - time.time()
                rate_status += f"🚫 Bloqueado: {remaining:.1f}s"
            else:
                rate_status += "✅ Normal"
        else:
            rate_status = "No disponible"
        
        embed.add_field(
            name="⚡ Rate Limiting",
            value=rate_status,
            inline=True
        )
        
        # Available commands info
        embed.add_field(
            name="💡 Comandos Disponibles",
            value="**Siempre disponibles:**\n"
                  "`$help` - Lista completa\n"
                  "`$ping` - Latencia\n"
                  "`$chat mensaje` - IA chat\n"
                  "`$play canción` - Música\n"
                  "`$health` - Este estado",
            inline=False
        )
        
        if getattr(bot, 'commands_synced', False):
            embed.add_field(
                name="⚡ Slash Commands",
                value="Usa `/help` para ver todos los comandos slash disponibles",
                inline=False
            )
        elif getattr(bot, 'emergency_mode', False):
            embed.add_field(
                name="🆘 Modo Emergencia",
                value="Solo comandos de texto disponibles. Admins pueden usar `$reset_emergency`",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error obteniendo estado del sistema: {e}")

@bot.command(name='force_sync')
@commands.has_permissions(administrator=True)
async def force_sync_admin(ctx):
    """Admin force sync using resilient architecture with reduced cooldown"""
    try:
        if getattr(bot, 'emergency_mode', False):
            await ctx.send("🚫 Bot en modo emergencia. Usar `$reset_emergency` primero.")
            return
        
        # Mark as admin-requested for reduced cooldown
        bot._admin_requested_sync = True
        
        status_msg = await ctx.send("🔄 Sincronizando comandos slash con protección admin...")
        
        # Check current state first
        if hasattr(bot, 'sync_circuit_breaker'):
            can_execute, reason = bot.sync_circuit_breaker.can_execute()
            if not can_execute:
                await status_msg.edit(content=f"🔒 Circuit breaker bloqueado: {reason}\nUsar `$reset_emergency` si es necesario.")
                return
        
        # Use the protected sync method
        success = await bot._attempt_resilient_sync()
        
        if success:
            embed = discord.Embed(
                title="✅ Comandos Slash Activados",
                description="Los comandos slash están ahora disponibles en Discord.\n"
                           "Escribe `/` para ver todos los comandos disponibles.",
                color=0x00FF00,
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Estado",
                value="🟢 Slash commands activos\n🟢 Rate limit respetado\n🟢 Circuit breaker funcionando",
                inline=False
            )
            await status_msg.edit(content="", embed=embed)
        else:
            # Get detailed status for debugging
            if hasattr(bot, 'rate_limiter'):
                can_request, wait_time = bot.rate_limiter.can_make_request("command_sync")
                if not can_request:
                    await status_msg.edit(content=f"⏰ Rate limit activo. Esperar {wait_time:.1f}s antes del próximo intento.")
                else:
                    await status_msg.edit(content="❌ Sincronización falló. Revisar logs para detalles.")
            else:
                await status_msg.edit(content="❌ Sincronización falló. Revisar logs para detalles.")
        
        # Clean up admin flag
        if hasattr(bot, '_admin_requested_sync'):
            delattr(bot, '_admin_requested_sync')
    
    except Exception as e:
        await ctx.send(f"❌ Error en force sync: {e}")
        # Clean up admin flag on error
        if hasattr(bot, '_admin_requested_sync'):
            delattr(bot, '_admin_requested_sync')

@bot.command(name='reset_emergency')
@commands.has_permissions(administrator=True)
async def reset_emergency_command(ctx):
    """Reset emergency mode and circuit breaker (admin only)"""
    try:
        if not getattr(bot, 'emergency_mode', False):
            await ctx.send("ℹ️ Bot no está en modo emergencia.")
            return
        
        # Reset emergency mode and circuit breaker
        bot.emergency_mode = False
        if hasattr(bot, 'sync_circuit_breaker'):
            bot.sync_circuit_breaker = CircuitBreaker()  # Reset circuit breaker
        
        await ctx.send("🔄 Modo emergencia reseteado. Intentando sincronizar...")
        
        await asyncio.sleep(2)
        success = await bot._attempt_resilient_sync()
        
        if success:
            await ctx.send("✅ ¡Sistema recuperado! Comandos slash activos.")
        else:
            await ctx.send("⚠️ Sincronización aún falla. Sistema en recuperación.")
    
    except Exception as e:
        await ctx.send(f"❌ Error reseteando modo emergencia: {e}")

@bot.command(name='circuit_status')
@commands.has_permissions(administrator=True)
async def circuit_status_command(ctx):
    """Show detailed circuit breaker status (admin only)"""
    try:
        if not hasattr(bot, 'sync_circuit_breaker'):
            await ctx.send("❌ Circuit breaker no disponible")
            return
        
        cb = bot.sync_circuit_breaker
        embed = discord.Embed(
            title="🛡️ Estado del Circuit Breaker",
            color=0xFFB6C1,
            timestamp=datetime.now()
        )
        
        # State info
        state_colors = {
            CircuitBreakerState.CLOSED: "🟢",
            CircuitBreakerState.OPEN: "🔴",
            CircuitBreakerState.HALF_OPEN: "🟡"
        }
        
        embed.add_field(
            name="Estado Actual",
            value=f"{state_colors.get(cb.state, '⚪')} {cb.state.value.title()}",
            inline=True
        )
        
        embed.add_field(
            name="Contadores",
            value=f"Fallos: {cb.failure_count}/{cb.failure_threshold}\n"
                  f"Éxitos: {cb.success_count}",
            inline=True
        )
        
        # Timing info
        if cb.last_failure_time > 0:
            last_failure = datetime.fromtimestamp(cb.last_failure_time)
            embed.add_field(
                name="Último Fallo",
                value=last_failure.strftime("%H:%M:%S"),
                inline=True
            )
        
        # Can execute info
        can_execute, reason = cb.can_execute()
        embed.add_field(
            name="¿Puede Ejecutar?",
            value=f"{'✅' if can_execute else '❌'} {reason}",
            inline=False
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error obteniendo estado circuit breaker: {e}")

# Simple force sync command fallback (keeping original structure)
@bot.command(name='force_sync_old')
@commands.has_permissions(administrator=True)
async def force_sync_commands(ctx):
    """DEPRECATED - Use $force_sync instead"""
    await ctx.send("⚠️ Comando deprecado. Usa `$force_sync` para sincronización protegida.")

# Emergency diagnostic command
@bot.command(name='diagnostic')
@commands.has_permissions(administrator=True)
async def diagnostic_command(ctx):
    """Run comprehensive diagnostic of rate limiting issues"""
    try:
        async with ctx.typing():
            # Run the diagnostic tool
            diagnostic_result = []
            
            # Check current state
            diagnostic_result.append("🔍 **DIAGNÓSTICO DEL SISTEMA RESILIENTE**")
            diagnostic_result.append("=" * 40)
            
            # Circuit breaker status
            if hasattr(bot, 'sync_circuit_breaker'):
                cb = bot.sync_circuit_breaker
                diagnostic_result.append(f"🛡️ Circuit Breaker: {cb.state.value} (fallos: {cb.failure_count})")
                can_execute, reason = cb.can_execute()
                diagnostic_result.append(f"   Puede ejecutar: {'✅' if can_execute else '❌'} {reason}")
            else:
                diagnostic_result.append("❌ Circuit Breaker: No inicializado")
            
            # Rate limiter status
            if hasattr(bot, 'rate_limiter'):
                rl = bot.rate_limiter
                current_time = time.time()
                is_blocked = current_time < rl.blocked_until
                diagnostic_result.append(f"⚡ Rate Limiter: {'🚫 Bloqueado' if is_blocked else '✅ Normal'}")
                diagnostic_result.append(f"   Requests actuales: {len(rl.global_requests)}/{rl.global_limit}")
                if is_blocked:
                    remaining = rl.blocked_until - current_time
                    diagnostic_result.append(f"   Tiempo restante: {remaining:.1f}s")
            else:
                diagnostic_result.append("❌ Rate Limiter: No inicializado")
            
            # Command sync status
            diagnostic_result.append(f"📝 Comandos Synced: {'✅' if getattr(bot, 'commands_synced', False) else '❌'}")
            diagnostic_result.append(f"🚨 Modo Emergencia: {'✅' if getattr(bot, 'emergency_mode', False) else '❌'}")
            diagnostic_result.append(f"🔄 Sync en Progreso: {'✅' if getattr(bot, 'sync_in_progress', False) else '❌'}")
            
            # Last sync attempt
            if hasattr(bot, 'last_sync_attempt') and bot.last_sync_attempt > 0:
                last_attempt = datetime.fromtimestamp(bot.last_sync_attempt)
                diagnostic_result.append(f"⏰ Último Intento: {last_attempt.strftime('%H:%M:%S')}")
            
            # Recommendations
            diagnostic_result.append("")
            diagnostic_result.append("💡 **RECOMENDACIONES:**")
            
            if getattr(bot, 'emergency_mode', False):
                diagnostic_result.append("- Usar `$reset_emergency` para salir del modo emergencia")
            elif not getattr(bot, 'commands_synced', False):
                diagnostic_result.append("- El sistema está esperando que expire el rate limit")
                diagnostic_result.append("- Los comandos de texto están disponibles mientras tanto")
            else:
                diagnostic_result.append("- Sistema funcionando correctamente")
            
            # System health
            diagnostic_result.append("")
            diagnostic_result.append("📊 **ESTADO DEL SISTEMA:**")
            diagnostic_result.append(f"- Latencia: {round(bot.latency * 1000)}ms")
            diagnostic_result.append(f"- Servidores: {len(bot.guilds)}")
            diagnostic_result.append(f"- Wavelink Nodes: {len(getattr(bot, 'connected_nodes', []))}")
            
            # Send results
            result_text = "\n".join(diagnostic_result)
            
            embed = discord.Embed(
                title="🔍 Diagnóstico del Sistema",
                description=f"```\n{result_text}\n```",
                color=0xFFB6C1,
                timestamp=datetime.now()
            )
            
            await ctx.send(embed=embed)
            
    except Exception as e:
        await ctx.send(f"❌ Error ejecutando diagnóstico: {e}")

# ================================================================
# COMANDOS SLASH CON SOPORTE COMPLETO PARA DMs 
# ================================================================

def detect_interaction_context(interaction: discord.Interaction) -> dict:
    """Detecta el contexto exacto de una interacción"""
    context = {
        'type': 'Desconocido',
        'channel': 'N/A',
        'guild': 'N/A',
        'is_dm': False,
        'is_private': False,
        'is_guild': False
    }
    
    # DM directo con el bot
    if interaction.channel and interaction.channel.type == discord.ChannelType.private:
        context.update({
            'type': 'DM con Bot',
            'channel': f'DM con {interaction.user.name}',
            'guild': None,
            'is_dm': True,
            'is_private': True
        })
    # Canal de servidor
    elif interaction.guild:
        context.update({
            'type': 'Servidor (Guild)',
            'channel': f'#{interaction.channel.name}' if interaction.channel else 'Canal desconocido',
            'guild': interaction.guild.name,
            'is_guild': True
        })
    # DM entre usuarios (private channel)
    elif not interaction.guild and interaction.channel:
        context.update({
            'type': 'DM entre Usuarios',
            'channel': 'Canal privado',
            'guild': None,
            'is_private': True
        })
    
    return context







# ====================================
# REDIS CACHE MANAGEMENT COMMANDS
# ====================================





# ====================================
# TICKET SYSTEM - ADMINISTRATIVE COMMANDS
# ====================================

class TicketSetupView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=60)
        
    @discord.ui.button(label="🌸 Configurar Mensaje Kawaii", style=discord.ButtonStyle.primary)
    async def setup_ticket_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = TicketSetupModal()
        await interaction.response.send_modal(modal)

class TicketSetupModal(discord.ui.Modal, title="🌸 Configuración de Tickets~ uwu"):
    def __init__(self):
        super().__init__()

    title_input = discord.ui.TextInput(
        label="💫 Título del mensaje",
        placeholder="🌸 Ticket de Soporte~ uwu",
        default="🌸 Ticket de Soporte~ uwu",
        max_length=100,
        required=False
    )
    
    description_input = discord.ui.TextInput(
        label="📝 Descripción kawaii",
        placeholder="💫 *susurra* Usa el botón de abajo para crear un ticket~ necesito ayudarte >.<",
        default="💫 *susurra* Usa el botón de abajo para crear un ticket~ necesito ayudarte >.<",
        style=discord.TextStyle.paragraph,
        max_length=1000,
        required=False
    )
    
    footer_input = discord.ui.TextInput(
        label="🐶 Pie de página",
        placeholder="🌙 *timidamente* Solo puedes tener 1 ticket abierto a la vez~ uwu",
        default="🌙 *timidamente* Solo puedes tener 1 ticket abierto a la vez~ uwu",
        max_length=200,
        required=False
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        
        embed = discord.Embed(
            title=self.title_input.value or "🌸 Ticket de Soporte~ uwu",
            description=self.description_input.value or "💫 *susurra* Usa el botón de abajo para crear un ticket~ necesito ayudarte >.<",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        
        if self.footer_input.value:
            embed.set_footer(text=self.footer_input.value)
        
        view = TicketCreateView()
        await interaction.followup.send(embeds=[embed], view=view)

class TicketCreateView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        
    @discord.ui.button(label="🌸 Crear ticket kawaii~", style=discord.ButtonStyle.success, custom_id="create_ticket")
    async def create_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_ticket_creation(interaction)
    
    async def handle_ticket_creation(self, interaction: discord.Interaction):
        # Check if user already has a ticket
        existing_ticket = None
        for channel in interaction.guild.channels:
            if channel.name.startswith(f"ticket-{interaction.user.name.lower()}"):
                existing_ticket = channel
                break
        
        if existing_ticket:
            embed = discord.Embed(
                title="🎫 Ya tienes un ticket UwU",
                description=f"*susurra tímidamente* Ya tienes un ticket activo en {existing_ticket.mention}~ uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        # Create ticket channel
        try:
            overwrites = {
                interaction.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                interaction.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                interaction.guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True)
            }
            
            # Add moderator roles if they exist
            for role in interaction.guild.roles:
                if any(keyword in role.name.lower() for keyword in ["mod", "admin", "staff"]):
                    overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True)
            
            ticket_channel = await interaction.guild.create_text_channel(
                name=f"ticket-{interaction.user.name.lower()}",
                overwrites=overwrites,
                topic=f"Ticket de soporte para {interaction.user.display_name}"
            )
            
            # Welcome embed
            welcome_embed = discord.Embed(
                title="🌸 ¡Ticket creado exitosamente!~ uwu",
                description=f"*celebra tímidamente* ¡Hola {interaction.user.mention}! Este es tu ticket personal~ >.<\n\n📝 Describe tu problema y un moderador te ayudará pronto uwu",
                color=COLORES_KAWAII["EXITO_KAWAII"]
            )
            welcome_embed.add_field(
                name="🔒 Cerrar ticket",
                value="Usa el botón de abajo cuando tu problema esté resuelto~ uwu",
                inline=False
            )
            
            view = TicketControlView()
            await ticket_channel.send(embed=welcome_embed, view=view)
            
            # Response to user
            success_embed = discord.Embed(
                title="✨ Ticket creado uwu",
                description=f"*susurra feliz* Tu ticket fue creado en {ticket_channel.mention}~ ¡Revísalo pronto! >.<",
                color=COLORES_KAWAII["EXITO_KAWAII"]
            )
            await interaction.response.send_message(embed=success_embed, ephemeral=True)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="😢 Error creando ticket",
                description="*llora suavemente* No pude crear tu ticket... ¿el bot tiene permisos? uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            logger.error(f"Error creating ticket: {e}")

class TicketControlView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        
    @discord.ui.button(label="🔒 Cerrar Ticket", style=discord.ButtonStyle.danger, custom_id="close_ticket")
    async def close_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.channel.name.startswith("ticket-"):
            await interaction.response.send_message("🎫 *se esconde* Este comando solo funciona en canales de tickets... uwu", ephemeral=True)
            return
        
        embed = discord.Embed(
            title="🔒 Cerrando ticket...",
            description="*susurra* El ticket se cerrará en 5 segundos... uwu",
            color=COLORES_KAWAII["ALERTA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        
        await asyncio.sleep(5)
        
        try:
            await interaction.channel.delete(reason=f"Ticket cerrado por {interaction.user}")
        except:
            pass

@bot.tree.command(name="ticket", description="🌸 *susurra* comandos de tickets~ uwu")
@app_commands.describe(
    accion="Acción del ticket",
    canal="Canal donde configurar (solo para setup)",
    limite="Límite de tickets (solo para limit)",
    usuario="Usuario para agregar/remover (solo para add/remove)"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="setup - ✨ configurar sistema", value="setup"),
    app_commands.Choice(name="close - 🔒 cerrar ticket actual", value="close"),
    app_commands.Choice(name="closeall - 🌙 cerrar todos los tickets", value="closeall"),
    app_commands.Choice(name="add - ➕ agregar usuario al ticket", value="add"),
    app_commands.Choice(name="remove - ➖ remover usuario del ticket", value="remove")
])
async def ticket_command(interaction: discord.Interaction, accion: str, canal: discord.TextChannel = None, limite: int = None, usuario: discord.User = None):
    if not interaction.user.guild_permissions.manage_guild:
        embed = discord.Embed(
            title="🔐 Sin permisos UwU",
            description="*se esconde* Necesitas permisos de `Gestionar Servidor`... lo siento >.<",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if accion == "setup":
        if not canal:
            embed = discord.Embed(
                title="📺 Canal requerido UwU",
                description="*susurra* Necesito que especifiques un canal para el setup... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not interaction.guild.me.guild_permissions.manage_channels:
            embed = discord.Embed(
                title="🥺 Sin permisos",
                description="*se disculpa* No tengo permisos para crear canales de tickets... lo siento uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        embed = discord.Embed(
            title="🌸 Configuración de Tickets~ uwu",
            description="*susurra tímidamente* Haz clic en el botón para configurar el mensaje de tickets~ uwu",
            color=COLORES_KAWAII["ROSA_KAWAII"]
        )
        view = TicketSetupView()
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        
    elif accion == "close":
        if not interaction.channel.name.startswith("ticket-"):
            embed = discord.Embed(
                title="🎫 No es un ticket",
                description="*se esconde* Este comando solo funciona en canales de tickets... uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        embed = discord.Embed(
            title="🔒 Cerrando ticket...",
            description="*susurra* El ticket se cerrará en 5 segundos... uwu",
            color=COLORES_KAWAII["ALERTA_KAWAII"]
        )
        await interaction.response.send_message(embed=embed)
        
        await asyncio.sleep(5)
        try:
            await interaction.channel.delete(reason=f"Ticket cerrado por {interaction.user}")
        except:
            pass
            
    elif accion == "closeall":
        closed = 0
        failed = 0
        
        for channel in interaction.guild.channels:
            if channel.name.startswith("ticket-"):
                try:
                    await channel.delete(reason=f"Tickets cerrados masivamente por {interaction.user}")
                    closed += 1
                except:
                    failed += 1
        
        embed = discord.Embed(
            title="🌙 Tickets cerrados",
            description=f"*informa tímidamente* ¡Terminado! Exitosos: `{closed}` Fallidos: `{failed}` uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "add":
        if not interaction.channel.name.startswith("ticket-"):
            embed = discord.Embed(
                title="🎫 No es un ticket",
                description="*se esconde* Este comando solo funciona en canales de tickets... uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not usuario:
            embed = discord.Embed(
                title="👥 Usuario requerido",
                description="*timidamente* Necesito que especifiques qué usuario agregar... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        try:
            await interaction.channel.set_permissions(usuario, read_messages=True, send_messages=True)
            embed = discord.Embed(
                title="✨ Usuario agregado",
                description=f"*celebra tímidamente* ¡{usuario.mention} agregado al ticket!~ uwu",
                color=COLORES_KAWAII["EXITO_KAWAII"]
            )
            await interaction.response.send_message(embed=embed)
        except:
            embed = discord.Embed(
                title="🥺 Error",
                description="*se disculpa* No pude agregar el usuario... ¿permisos? >.<",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
    elif accion == "remove":
        if not interaction.channel.name.startswith("ticket-"):
            embed = discord.Embed(
                title="🎫 No es un ticket",
                description="*se esconde* Este comando solo funciona en canales de tickets... uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not usuario:
            embed = discord.Embed(
                title="👥 Usuario requerido",
                description="*murmura* Dime qué usuario quitar del ticket... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        try:
            await interaction.channel.set_permissions(usuario, read_messages=False, send_messages=False)
            embed = discord.Embed(
                title="✨ Usuario removido",
                description=f"*susurra* {usuario.mention} removido del ticket~ uwu",
                color=COLORES_KAWAII["EXITO_KAWAII"]
            )
            await interaction.response.send_message(embed=embed)
        except:
            embed = discord.Embed(
                title="🥺 Error",
                description="*se disculpa* No pude remover el usuario... ¿permisos? >.<",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

# ====================================
# WELCOME & FAREWELL SYSTEM
# ====================================

# In-memory storage for welcome/farewell settings
welcome_settings = {}
farewell_settings = {}

def get_welcome_settings(guild_id):
    if guild_id not in welcome_settings:
        welcome_settings[guild_id] = {
            "enabled": False,
            "channel": None,
            "embed": {
                "description": "¡Bienvenido/a {user}! Espero que disfrutes tu estadía aquí~ uwu 💖",
                "thumbnail": True,
                "color": "#FFB6C1",
                "footer": "🌸 Bienvenido/a al servidor~ uwu",
                "image": None
            }
        }
    return welcome_settings[guild_id]

def get_farewell_settings(guild_id):
    if guild_id not in farewell_settings:
        farewell_settings[guild_id] = {
            "enabled": False,
            "channel": None,
            "embed": {
                "description": "*se despide tímidamente* ¡Adiós {user}! Te extrañaremos~ >.<",
                "thumbnail": True,
                "color": "#DDA0DD",
                "footer": "🌙 Hasta la vista~ uwu",
                "image": None
            }
        }
    return farewell_settings[guild_id]

def is_hex_color(color_string):
    if color_string.startswith('#'):
        color_string = color_string[1:]
    try:
        int(color_string, 16)
        return len(color_string) == 6
    except ValueError:
        return False

async def build_greeting_embed(member, greeting_type, settings):
    """Build greeting embed for welcome or farewell"""
    embed_settings = settings["embed"]
    
    if greeting_type == "WELCOME":
        title = "🌸 ¡Nueva personita!~ uwu"
        color = int(embed_settings.get("color", "#FFB6C1").replace("#", ""), 16)
    else:  # FAREWELL
        title = "🌙 Despedida kawaii~ >.<"
        color = int(embed_settings.get("color", "#DDA0DD").replace("#", ""), 16)
    
    description = embed_settings["description"].replace("{user}", member.mention).replace("{server}", member.guild.name)
    
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )
    
    if embed_settings.get("thumbnail", True):
        embed.set_thumbnail(url=member.display_avatar.url)
    
    if embed_settings.get("image"):
        embed.set_image(url=embed_settings["image"])
    
    if embed_settings.get("footer"):
        embed.set_footer(text=embed_settings["footer"])
    
    embed.add_field(name="👤 Usuario", value=member.mention, inline=True)
    embed.add_field(name="🆔 ID", value=member.id, inline=True)
    embed.add_field(name="📅 Cuenta creada", value=member.created_at.strftime("%d/%m/%Y"), inline=True)
    
    if greeting_type == "WELCOME":
        embed.add_field(name="👥 Miembro #", value=member.guild.member_count, inline=True)
    
    return embed

@bot.event
async def on_member_join(member):
    """Handle member join for welcome messages"""
    settings = get_welcome_settings(member.guild.id)
    
    if not settings["enabled"] or not settings["channel"]:
        return
    
    channel = bot.get_channel(settings["channel"])
    if not channel:
        return
    
    try:
        embed = await build_greeting_embed(member, "WELCOME", settings)
        await channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")

@bot.event
async def on_member_remove(member):
    """Handle member leave for farewell messages"""
    settings = get_farewell_settings(member.guild.id)
    
    if not settings["enabled"] or not settings["channel"]:
        return
    
    channel = bot.get_channel(settings["channel"])
    if not channel:
        return
    
    try:
        embed = await build_greeting_embed(member, "FAREWELL", settings)
        await channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Error sending farewell message: {e}")

@bot.tree.command(name="welcome", description="🌸 *susurra* configurar mensaje de bienvenida~ uwu")
@app_commands.describe(
    accion="Configuración de bienvenida",
    canal="Canal para mensajes de bienvenida",
    estado="Activar o desactivar",
    contenido="Contenido del mensaje",
    color="Color hexadecimal (ej: #FFB6C1)",
    url="URL de imagen"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="status - 🌸 activar/desactivar", value="status"),
    app_commands.Choice(name="channel - 💫 configurar canal", value="channel"),
    app_commands.Choice(name="preview - ✨ ver preview", value="preview"),
    app_commands.Choice(name="desc - 📝 descripción", value="desc"),
    app_commands.Choice(name="thumbnail - 🖼️ miniatura on/off", value="thumbnail"),
    app_commands.Choice(name="color - 🎨 color del embed", value="color"),
    app_commands.Choice(name="footer - 👣 pie de página", value="footer"),
    app_commands.Choice(name="image - 🖼️ imagen del embed", value="image")
])
@app_commands.choices(estado=[
    app_commands.Choice(name="ON", value="ON"),
    app_commands.Choice(name="OFF", value="OFF")
])
async def welcome_command(interaction: discord.Interaction, accion: str, canal: discord.TextChannel = None, estado: str = None, contenido: str = None, color: str = None, url: str = None):
    if not interaction.user.guild_permissions.manage_guild:
        embed = discord.Embed(
            title="🔐 Sin permisos UwU",
            description="*se esconde* Necesitas permisos de `Gestionar Servidor`... lo siento >.<",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    settings = get_welcome_settings(interaction.guild.id)
    
    if accion == "status":
        if not estado:
            embed = discord.Embed(
                title="🥺 Estado requerido",
                description="*se confunde* Necesito que me digas ON o OFF... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        enabled = estado.upper() == "ON"
        settings["enabled"] = enabled
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Configuración guardada",
            description=f"*celebra suavemente* ¡Mensajes de bienvenida {'habilitados' if enabled else 'deshabilitados'}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "channel":
        if not canal:
            embed = discord.Embed(
                title="📺 Canal requerido",
                description="*timidamente* Necesito que especifiques un canal... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not canal.permissions_for(interaction.guild.me).send_messages or not canal.permissions_for(interaction.guild.me).embed_links:
            embed = discord.Embed(
                title="🥺 Sin permisos",
                description=f"*se disculpa* No puedo enviar mensajes a {canal.mention}... necesito permisos uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["channel"] = canal.id
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Canal configurado",
            description=f"*susurra feliz* ¡Los mensajes de bienvenida se enviarán a {canal.mention}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "preview":
        if not settings["enabled"]:
            embed = discord.Embed(
                title="🌸 Sistema desactivado",
                description="*susurra* El mensaje de bienvenida no está habilitado... actívalo primero uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not settings["channel"]:
            embed = discord.Embed(
                title="📺 Sin canal",
                description="*mira nerviosa* No hay canal configurado para mensajes de bienvenida... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        target_channel = bot.get_channel(settings["channel"])
        if not target_channel:
            embed = discord.Embed(
                title="📺 Canal no encontrado",
                description="*se confunde* No encuentro el canal configurado... ¿fue eliminado? uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        preview_embed = await build_greeting_embed(interaction.user, "WELCOME", settings)
        await target_channel.send(content="🌸 **PREVIEW DE BIENVENIDA** 🌸", embed=preview_embed)
        
        embed = discord.Embed(
            title="✨ Preview enviado",
            description=f"*celebra tímidamente* ¡Preview enviado a {target_channel.mention}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "desc":
        if not contenido:
            embed = discord.Embed(
                title="📝 Contenido requerido",
                description="*timidamente* Necesito contenido para la descripción... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["description"] = contenido
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Descripción actualizada",
            description="*trabaja diligentemente* ¡Configuración guardada! Mensaje de bienvenida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "thumbnail":
        if not estado:
            embed = discord.Embed(
                title="🖼️ Estado requerido",
                description="*se esconde* Necesito que me digas ON o OFF... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        thumbnail_enabled = estado.upper() == "ON"
        settings["embed"]["thumbnail"] = thumbnail_enabled
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Miniatura configurada",
            description="*ajusta la imagen* ¡Configuración guardada! Mensaje de bienvenida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "color":
        if not color or not is_hex_color(color):
            embed = discord.Embed(
                title="🎨 Color inválido",
                description="*mira nerviosamente* Necesito un código hex válido (ej: #FFB6C1)... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["color"] = color
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Color actualizado",
            description="*pinta suavemente* ¡Configuración guardada! Mensaje de bienvenida actualizado~ uwu",
            color=int(color.replace("#", ""), 16)
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "footer":
        if not contenido:
            embed = discord.Embed(
                title="👣 Contenido requerido",
                description="*susurra* Necesito contenido para el footer... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["footer"] = contenido
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Footer actualizado",
            description="*escribe tímidamente* ¡Configuración guardada! Mensaje de bienvenida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "image":
        if not url:
            embed = discord.Embed(
                title="🖼️ URL requerida",
                description="*se esconde tímidamente* Necesito una URL de imagen válida... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["image"] = url
        welcome_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Imagen configurada",
            description="*coloca imagen cuidadosamente* ¡Configuración guardada! Mensaje de bienvenida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="farewell", description="🌙 *susurra* configurar mensaje de despedida~ uwu")
@app_commands.describe(
    accion="Configuración de despedida",
    canal="Canal para mensajes de despedida",
    estado="Activar o desactivar",
    contenido="Contenido del mensaje",
    color="Color hexadecimal (ej: #DDA0DD)",
    url="URL de imagen"
)
@app_commands.choices(accion=[
    app_commands.Choice(name="status - 🌙 activar/desactivar", value="status"),
    app_commands.Choice(name="channel - 💫 configurar canal", value="channel"),
    app_commands.Choice(name="preview - ✨ ver preview", value="preview"),
    app_commands.Choice(name="desc - 📝 descripción", value="desc"),
    app_commands.Choice(name="thumbnail - 🖼️ miniatura on/off", value="thumbnail"),
    app_commands.Choice(name="color - 🎨 color del embed", value="color"),
    app_commands.Choice(name="footer - 👣 pie de página", value="footer"),
    app_commands.Choice(name="image - 🖼️ imagen del embed", value="image")
])
@app_commands.choices(estado=[
    app_commands.Choice(name="ON", value="ON"),
    app_commands.Choice(name="OFF", value="OFF")
])
async def farewell_command(interaction: discord.Interaction, accion: str, canal: discord.TextChannel = None, estado: str = None, contenido: str = None, color: str = None, url: str = None):
    if not interaction.user.guild_permissions.manage_guild:
        embed = discord.Embed(
            title="🔐 Sin permisos UwU",
            description="*se esconde* Necesitas permisos de `Gestionar Servidor`... lo siento >.<",
            color=COLORES_KAWAII["ERROR_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    settings = get_farewell_settings(interaction.guild.id)
    
    if accion == "status":
        if not estado:
            embed = discord.Embed(
                title="🥺 Estado requerido",
                description="*se confunde* Necesito que me digas ON o OFF... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        enabled = estado.upper() == "ON"
        settings["enabled"] = enabled
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Configuración guardada",
            description=f"*celebra suavemente* ¡Mensajes de despedida {'habilitados' if enabled else 'deshabilitados'}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "channel":
        if not canal:
            embed = discord.Embed(
                title="📺 Canal requerido",
                description="*timidamente* Necesito que especifiques un canal... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not canal.permissions_for(interaction.guild.me).send_messages or not canal.permissions_for(interaction.guild.me).embed_links:
            embed = discord.Embed(
                title="🥺 Sin permisos",
                description=f"*se disculpa* No puedo enviar mensajes a {canal.mention}... necesito permisos uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["channel"] = canal.id
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Canal configurado",
            description=f"*susurra feliz* ¡Los mensajes de despedida se enviarán a {canal.mention}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "preview":
        if not settings["enabled"]:
            embed = discord.Embed(
                title="🌙 Sistema desactivado",
                description="*susurra* El mensaje de despedida no está habilitado... actívalo primero uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        if not settings["channel"]:
            embed = discord.Embed(
                title="📺 Sin canal",
                description="*mira nerviosa* No hay canal configurado para mensajes de despedida... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        target_channel = bot.get_channel(settings["channel"])
        if not target_channel:
            embed = discord.Embed(
                title="📺 Canal no encontrado",
                description="*se confunde* No encuentro el canal configurado... ¿fue eliminado? uwu",
                color=COLORES_KAWAII["ERROR_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        preview_embed = await build_greeting_embed(interaction.user, "FAREWELL", settings)
        await target_channel.send(content="🌙 **PREVIEW DE DESPEDIDA** 🌙", embed=preview_embed)
        
        embed = discord.Embed(
            title="✨ Preview enviado",
            description=f"*celebra tímidamente* ¡Preview enviado a {target_channel.mention}!~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "desc":
        if not contenido:
            embed = discord.Embed(
                title="📝 Contenido requerido",
                description="*timidamente* Necesito contenido para la descripción... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["description"] = contenido
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Descripción actualizada",
            description="*trabaja diligentemente* ¡Configuración guardada! Mensaje de despedida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "thumbnail":
        if not estado:
            embed = discord.Embed(
                title="🖼️ Estado requerido",
                description="*se esconde* Necesito que me digas ON o OFF... >.<",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        thumbnail_enabled = estado.upper() == "ON"
        settings["embed"]["thumbnail"] = thumbnail_enabled
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Miniatura configurada",
            description="*ajusta la imagen* ¡Configuración guardada! Mensaje de despedida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "color":
        if not color or not is_hex_color(color):
            embed = discord.Embed(
                title="🎨 Color inválido",
                description="*mira nerviosamente* Necesito un código hex válido (ej: #DDA0DD)... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["color"] = color
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Color actualizado",
            description="*pinta suavemente* ¡Configuración guardada! Mensaje de despedida actualizado~ uwu",
            color=int(color.replace("#", ""), 16)
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "footer":
        if not contenido:
            embed = discord.Embed(
                title="👣 Contenido requerido",
                description="*susurra* Necesito contenido para el footer... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["footer"] = contenido
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Footer actualizado",
            description="*escribe tímidamente* ¡Configuración guardada! Mensaje de despedida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    elif accion == "image":
        if not url:
            embed = discord.Embed(
                title="🖼️ URL requerida",
                description="*se esconde tímidamente* Necesito una URL de imagen válida... uwu",
                color=COLORES_KAWAII["ALERTA_KAWAII"]
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        
        settings["embed"]["image"] = url
        farewell_settings[interaction.guild.id] = settings
        
        embed = discord.Embed(
            title="✨ Imagen configurada",
            description="*coloca imagen cuidadosamente* ¡Configuración guardada! Mensaje de despedida actualizado~ uwu",
            color=COLORES_KAWAII["EXITO_KAWAII"]
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

# ====================================
# VECTOR MEMORY & POSTGRESQL COMMANDS
# ====================================







# Auto-sync disabled to prevent rate limit issues
# Commands are now primarily available via prefix ($command)

# Add startup time tracking to bot initialization
if not hasattr(bot, 'startup_time'):
    bot.startup_time = time.time()

# Main function to run the bot
async def main():
    """Main function to run the bot"""
    discord_token = os.getenv('DISCORD_TOKEN')
    
    if not discord_token:
        logger.error("❌ DISCORD_TOKEN not found in environment variables")
        return
    
    # Initialize all managers
    bot.affection_manager = AffectionManager()
    bot.automod_manager = AutoModManager()
    bot.ai_provider = AIProvider()
    bot.gif_provider = GifProvider()
    bot.personality_manager = PersonalityManager()
    bot.search_manager = AdvancedSearchProvider()
    bot.postgresql_manager = PostgreSQLManager()
    
    # Initialize multimodal system
    await initialize_multimodal_system(bot.ai_provider)
    
    logger.info("✅ All managers initialized")
    
    try:
        logger.info("🚀 Starting Sakura IA bot...")
        await bot.start(discord_token)
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    asyncio.run(main())

