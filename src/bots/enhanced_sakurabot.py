import discord
from discord.ext import commands
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
import logging
import time
from datetime import datetime
import re
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.pool = None
    
    def get_connection(self):
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
    
    async def save_conversation(self, user_id: int, user_msg: str, bot_response: str = None):
        """Save conversation to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if bot_response is None:
                        cur.execute(
                            "INSERT INTO conversations (user_id, user_msg) VALUES (%s, %s) RETURNING id",
                            (user_id, user_msg)
                        )
                        return cur.fetchone()['id']
                    else:
                        cur.execute(
                            "UPDATE conversations SET bot_response = %s WHERE user_id = %s AND user_msg = %s AND bot_response IS NULL ORDER BY timestamp DESC LIMIT 1",
                            (bot_response, user_id, user_msg)
                        )
                        conn.commit()
        except Exception as e:
            logger.error(f"Database error saving conversation: {e}")
    
    async def save_memory(self, user_id: int, key: str, value: str):
        """Save user memory data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory (user_id, key, value) VALUES (%s, %s, %s) ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                        (user_id, key, value)
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Database error saving memory: {e}")
    
    async def get_memory(self, user_id: int, key: str = None) -> Dict[str, str]:
        """Get user memory data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if key:
                        cur.execute("SELECT value FROM memory WHERE user_id = %s AND key = %s", (user_id, key))
                        result = cur.fetchone()
                        return {key: result['value']} if result else {}
                    else:
                        cur.execute("SELECT key, value FROM memory WHERE user_id = %s", (user_id,))
                        return {row['key']: row['value'] for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"Database error getting memory: {e}")
            return {}
    
    async def log_error(self, user_id: int, service: str, error_msg: str):
        """Log API errors"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO errors (user_id, service, error_msg) VALUES (%s, %s, %s)",
                        (user_id, service, error_msg)
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Database error logging error: {e}")

class EnsembleLLM:
    def __init__(self):
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('LLAMA_API_KEY')
        self.huggingface_key = os.getenv('HUGGINGFACE_API_KEY')
        self.db = DatabaseManager()
        self.timeout = 6
    
    async def call_openrouter(self, prompt: str, user_id: int) -> Optional[str]:
        """Call OpenRouter API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openrouter_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'meta-llama/llama-3.2-90b-vision-instruct',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1500,
                'temperature': 0.7
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post('https://openrouter.ai/api/v1/chat/completions', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_msg = f"OpenRouter API error: {response.status}"
                        await self.db.log_error(user_id, "openrouter", error_msg)
                        return None
        except Exception as e:
            await self.db.log_error(user_id, "openrouter", str(e))
            return None
    
    async def call_mistral(self, prompt: str, user_id: int) -> Optional[str]:
        """Call Mistral via OpenRouter"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openrouter_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'mistralai/mistral-7b-instruct',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1500,
                'temperature': 0.7
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post('https://openrouter.ai/api/v1/chat/completions', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_msg = f"Mistral API error: {response.status}"
                        await self.db.log_error(user_id, "mistral", error_msg)
                        return None
        except Exception as e:
            await self.db.log_error(user_id, "mistral", str(e))
            return None
    
    async def call_huggingface(self, prompt: str, user_id: int) -> Optional[str]:
        """Call HuggingFace API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.huggingface_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': 1500,
                    'temperature': 0.7,
                    'return_full_text': False
                }
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post('https://api-inference.huggingface.co/models/facebook/opt-1.3b', 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            return data[0].get('generated_text', '')
                        return str(data)
                    else:
                        error_msg = f"HuggingFace API error: {response.status}"
                        await self.db.log_error(user_id, "huggingface", error_msg)
                        return None
        except Exception as e:
            await self.db.log_error(user_id, "huggingface", str(e))
            return None
    
    def extract_best_phrase(self, response: str) -> str:
        """Extract the most coherent phrase from a response"""
        if not response:
            return ""
        
        # Split into sentences and find the most complete one
        sentences = re.split(r'[.!?]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return response.strip()
        
        # Score sentences by length and coherence
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            # Simple scoring: prefer longer, complete sentences
            score = len(sentence)
            if sentence.endswith(('.', '!', '?')):
                score += 10
            if any(word in sentence.lower() for word in ['hola', 'gracias', 'sakura', 'usuario']):
                score += 5
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence or sentences[0]
    
    async def ensemble_response(self, prompt: str, user_id: int) -> str:
        """Generate ensemble response from multiple LLM services"""
        # Call all services in parallel
        tasks = [
            self.call_openrouter(prompt, user_id),
            self.call_mistral(prompt, user_id),
            self.call_huggingface(prompt, user_id)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid responses
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, str) and response:
                valid_responses.append(response)
        
        if not valid_responses:
            return "Lo siento, hubo un problema con mis servicios de IA. Intenta de nuevo más tarde. — SakuraBot 🌸"
        
        # Extract best phrases and combine
        best_phrases = []
        for response in valid_responses:
            phrase = self.extract_best_phrase(response)
            if phrase and phrase not in best_phrases:
                best_phrases.append(phrase)
        
        # Combine phrases intelligently
        if len(best_phrases) == 1:
            final_response = best_phrases[0]
        else:
            # Join with appropriate connectors
            final_response = ". ".join(best_phrases[:2])  # Limit to 2 best phrases
        
        # Ensure proper ending
        if not final_response.endswith(('.', '!', '?')):
            final_response += "."
        
        return f"{final_response} — SakuraBot 🌸"

class SakuraBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
        self.db = DatabaseManager()
        self.llm = EnsembleLLM()
    
    async def setup_hook(self):
        """Setup hook called when bot starts"""
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} commands")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"{self.user} está conectado con sistema ensemble LLM!")
        logger.info(f"Bot está en {len(self.guilds)} servidores")
    
    def extract_important_data(self, message: str) -> Dict[str, str]:
        """Extract important data from user message"""
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
    
    async def process_triggers(self, message: discord.Message):
        """Process special triggers in user messages"""
        content = message.content.strip()
        user_id = message.author.id
        
        # Save initial conversation entry
        await self.db.save_conversation(user_id, content)
        
        # Extract and save important data
        important_data = self.extract_important_data(content)
        for key, value in important_data.items():
            await self.db.save_memory(user_id, key, value)
        
        response = None
        
        # Trigger 1: Message reply starting with $
        if message.reference and content.startswith('$'):
            action = content[1:].strip()
            if action:
                response = await self.llm.ensemble_response(f"El usuario quiere que hagas: {action}", user_id)
        
        # Trigger 2: @SakuraBot mention
        elif self.user.mentioned_in(message):
            mention_content = content.replace(f'<@{self.user.id}>', '').strip()
            if mention_content:
                response = await self.llm.ensemble_response(mention_content, user_id)
        
        # Trigger 3: "Sakura" followed by command
        elif content.lower().startswith('sakura '):
            command_part = content[7:].strip()
            if command_part:
                # Special commands
                if command_part.lower() in ['ia', 'chat', 'habla']:
                    response = "¡Hola! Soy SakuraBot, tu asistente IA con sistema ensemble. ¿En qué puedo ayudarte? — SakuraBot 🌸"
                elif command_part.lower().startswith('hug ') or command_part.lower().startswith('abrazo '):
                    target = command_part.split(' ', 1)[1] if ' ' in command_part else "alguien especial"
                    response = f"*abraza cariñosamente a {target}* ¡Abrazo kawaii enviado con amor! UwU — SakuraBot 🌸"
                elif command_part.lower().startswith('kiss ') or command_part.lower().startswith('beso '):
                    target = command_part.split(' ', 1)[1] if ' ' in command_part else "alguien especial"
                    response = f"*envía un besito volador a {target}* ¡Muah~ besito kawaii! 💖 — SakuraBot 🌸"
                else:
                    response = await self.llm.ensemble_response(f"Sakura, {command_part}", user_id)
        
        # General conversation if no specific trigger
        elif not message.author.bot:
            # Only respond to direct mentions or specific keywords
            sakura_keywords = ['sakura', 'bot', 'ayuda', 'help', 'hola', 'hello']
            if any(keyword in content.lower() for keyword in sakura_keywords):
                response = await self.llm.ensemble_response(content, user_id)
        
        # Send response if generated
        if response:
            await message.channel.send(response)
            await self.db.save_conversation(user_id, content, response)
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        if message.author.bot:
            return
        
        await self.process_triggers(message)
        await self.process_commands(message)

# Slash Commands
bot = SakuraBot()

@bot.tree.command(name="ensemble", description="Chat con SakuraBot usando sistema ensemble LLM")
async def ensemble_command(interaction: discord.Interaction, prompt: str):
    """Chat using ensemble LLM system"""
    await interaction.response.defer()
    
    user_id = interaction.user.id
    await bot.db.save_conversation(user_id, prompt)
    
    # Extract and save important data
    important_data = bot.extract_important_data(prompt)
    for key, value in important_data.items():
        await bot.db.save_memory(user_id, key, value)
    
    response = await bot.llm.ensemble_response(prompt, user_id)
    
    embed = discord.Embed(
        title="🌸 SakuraBot Ensemble Response",
        description=response,
        color=0xFFB6C1
    )
    
    await interaction.followup.send(embed=embed)
    await bot.db.save_conversation(user_id, prompt, response)

@bot.tree.command(name="memoria", description="Ver tu memoria guardada con SakuraBot")
async def memoria_command(interaction: discord.Interaction):
    """View user's saved memory"""
    user_id = interaction.user.id
    memory_data = await bot.db.get_memory(user_id)
    
    if not memory_data:
        embed = discord.Embed(
            title="💭 Memoria Vacía",
            description="No tengo datos guardados sobre ti aún. ¡Háblame más para que pueda recordarte! — SakuraBot 🌸",
            color=0xFFB6C1
        )
    else:
        embed = discord.Embed(
            title="💭 Tu Memoria Conmigo",
            description="Esto es lo que recuerdo sobre ti:",
            color=0xFFB6C1
        )
        
        for key, value in memory_data.items():
            embed.add_field(name=key.title(), value=value, inline=False)
        
        embed.set_footer(text="— SakuraBot 🌸")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="status", description="Estado del sistema ensemble de SakuraBot")
async def status_command(interaction: discord.Interaction):
    """Check ensemble system status"""
    embed = discord.Embed(
        title="🤖 Estado del Sistema Ensemble",
        description="Estado de los servicios LLM",
        color=0xFFB6C1
    )
    
    # Check API keys
    services = {
        "OpenRouter/Llama": "✅ Configurado" if bot.llm.openrouter_key else "❌ Sin API Key",
        "HuggingFace": "✅ Configurado" if bot.llm.huggingface_key else "❌ Sin API Key",
        "Base de Datos": "✅ Conectado" if bot.db.database_url else "❌ Sin conexión"
    }
    
    for service, status in services.items():
        embed.add_field(name=service, value=status, inline=True)
    
    embed.add_field(
        name="🧠 Sistema Ensemble",
        value="3 LLMs trabajando en paralelo:\n• OpenRouter (Llama)\n• Mistral\n• HuggingFace (OPT)",
        inline=False
    )
    
    embed.set_footer(text="— SakuraBot 🌸")
    await interaction.response.send_message(embed=embed)

# Run the bot
if __name__ == "__main__":
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("DISCORD_TOKEN not found!")
    else:
        bot.run(token)