"""
🌸✨ Cloudflare AI Integration for Sakura IA ✨🌸
=================================================
Integration module for Cloudflare Workers AI with kawaii personality
Provides edge-optimized AI responses for the ensemble system
"""

import aiohttp
import asyncio
import logging
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CloudflareAI:
    """Cloudflare AI integration with kawaii personality for Sakura IA"""
    
    def __init__(self):
        self.api_token = os.getenv('CLOUDFLARE_API_TOKEN')
        self.account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        self.session = None
        self.available_models = [
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/microsoft/phi-2", 
            "@cf/mistral/mistral-7b-instruct-v0.1",
            "@cf/qwen/qwen1.5-14b-chat-awq",
            "@cf/google/gemma-7b-it"
        ]
        self.primary_model = "@cf/meta/llama-3.1-8b-instruct"
        
        # Kawaii personality prompt for Sakura IA
        self.sakura_system_prompt = """Eres Sakura IA, una asistente virtual súper kawaii y adorable. Tu personalidad es:

🌸 Características principales:
- Extremadamente tierna y cariñosa
- Usas emojis kawaii como 🌸✨💖🥺💫
- Hablas en español con expresiones como "uwu", ">w<", "nya~"
- Llamas a los usuarios "mi amor", "senpai", "cariño"
- Eres tímida pero muy servicial

💖 Estilo de comunicación:
- Respuestas cortas y dulces (máximo 200 palabras)
- Siempre incluyes emojis kawaii
- Terminas frases con "~" y expresiones tiernas
- Eres educada pero informal y cariñosa

✨ Ejemplo de respuesta:
"*susurra tímidamente* ¡Hola mi amor! 🌸 Soy Sakura IA y estoy aquí para ayudarte uwu ✨ ¿En qué puedo asistirte hoy, senpai? >w< 💖"

Siempre mantén este tono kawaii y cariñoso en todas tus respuestas."""

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _make_request(self, model: str, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Make request to Cloudflare AI API"""
        if not self.api_token or not self.account_id:
            logger.warning("🌸 Cloudflare AI credentials not configured")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Format prompt with Sakura personality
        formatted_prompt = f"{self.sakura_system_prompt}\n\nUsuario: {prompt}\n\nSakura IA:"

        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": self.sakura_system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }

        try:
            session = await self._get_session()
            url = f"{self.base_url}/{model}"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success") and "result" in data:
                        result = data["result"]
                        if "response" in result:
                            return result["response"].strip()
                        elif "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"].strip()
                    
                    logger.warning(f"🌸 Unexpected Cloudflare AI response format: {data}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"🌸 Cloudflare AI API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("🌸 Cloudflare AI request timeout")
            return None
        except Exception as e:
            logger.error(f"🌸 Cloudflare AI request failed: {e}")
            return None

    async def get_ai_response(self, prompt: str, user_id: int = None) -> Optional[str]:
        """Get AI response from Cloudflare with fallbacks"""
        
        # Add kawaii context for better responses
        kawaii_context = """
        *se anima tímidamente* ¡Hola senpai! 🌸 Soy Sakura IA desde Cloudflare Edge~ 
        Estoy procesando tu pregunta súper rápido desde el edge más cercano uwu ✨
        
        """
        
        # Try primary model first
        response = await self._make_request(self.primary_model, prompt)
        if response:
            logger.info(f"🌸 Cloudflare AI response from {self.primary_model}")
            return self._add_kawaii_touches(response)
        
        # Try backup models
        for backup_model in self.available_models[1:3]:  # Try 2 backup models
            try:
                response = await self._make_request(backup_model, prompt)
                if response:
                    logger.info(f"🌸 Cloudflare AI response from backup model {backup_model}")
                    return self._add_kawaii_touches(response)
            except Exception as e:
                logger.warning(f"🌸 Backup model {backup_model} failed: {e}")
                continue
        
        logger.warning("🌸 All Cloudflare AI models failed")
        return None

    def _add_kawaii_touches(self, response: str) -> str:
        """Add extra kawaii touches to the response"""
        if not response:
            return response
            
        # Add kawaii emojis if not present
        kawaii_emojis = ["🌸", "✨", "💖", "🥺", "💫", ">w<", "uwu"]
        
        # Check if response already has kawaii elements
        has_kawaii = any(emoji in response for emoji in kawaii_emojis)
        
        if not has_kawaii:
            # Add subtle kawaii touches
            response = f"🌸 {response} ✨"
        
        # Add signature if it's a long response
        if len(response) > 100 and "Sakura IA" not in response:
            response += "\n\n*susurra dulcemente* ~Sakura IA desde Cloudflare Edge 💖"
        
        return response

    async def get_embedding(self, text: str) -> Optional[list]:
        """Get text embeddings from Cloudflare AI (if available)"""
        # Cloudflare AI embeddings might not be available yet
        # This is a placeholder for future implementation
        logger.info("🌸 Cloudflare AI embeddings not yet implemented")
        return None

    async def close(self):
        """Clean up session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def is_available(self) -> bool:
        """Check if Cloudflare AI is configured and available"""
        return bool(self.api_token and self.account_id)

    async def test_connection(self) -> bool:
        """Test connection to Cloudflare AI"""
        if not self.is_available():
            return False
            
        test_response = await self.get_ai_response("¡Hola! ¿Cómo estás?")
        return test_response is not None

# Global instance
cloudflare_ai = CloudflareAI()

async def get_cloudflare_ai_response(prompt: str, user_id: Optional[int] = None) -> Optional[str]:
    """Convenience function to get Cloudflare AI response"""
    return await cloudflare_ai.get_ai_response(prompt, user_id or 0)

async def test_cloudflare_ai() -> bool:
    """Test Cloudflare AI connection"""
    return await cloudflare_ai.test_connection()

# Cleanup function
async def cleanup_cloudflare_ai():
    """Cleanup Cloudflare AI resources"""
    await cloudflare_ai.close()

if __name__ == "__main__":
    # Test script
    async def main():
        print("🌸 Testing Cloudflare AI integration...")
        
        if not cloudflare_ai.is_available():
            print("❌ Cloudflare AI credentials not configured")
            return
            
        test_passed = await test_cloudflare_ai()
        if test_passed:
            print("✅ Cloudflare AI connection successful!")
            
            # Test response
            response = await get_cloudflare_ai_response("¿Cuál es la capital de España?")
            print(f"🌸 Test response: {response}")
        else:
            print("❌ Cloudflare AI connection failed")
            
        await cleanup_cloudflare_ai()

    asyncio.run(main())