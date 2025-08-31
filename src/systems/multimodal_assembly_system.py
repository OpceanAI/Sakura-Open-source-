"""
🌸✨ Sistema de Ensamblaje Multimodal Sakura IA ✨🌸
================================================
Sistema avanzado para detectar y ensamblar contenido de audio, texto e imagen
Procesamiento inteligente con múltiples proveedores de IA
"""

import os
import asyncio
import logging
import discord
import aiohttp
import json
import base64
import io
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from gtts import gTTS
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    sr = None
    HAS_SPEECH_RECOGNITION = False
import tempfile
import pathlib
try:
    import wave
    HAS_WAVE = True
except ImportError:
    wave = None
    HAS_WAVE = False

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Tipos de contenido detectables"""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"

@dataclass
class MediaContent:
    """Representación de contenido multimedia"""
    content_type: ContentType
    data: bytes
    metadata: Dict[str, Any]
    analysis_result: Optional[str] = None
    confidence: float = 0.0

@dataclass
class MultimodalAssembly:
    """Ensamblaje de contenido multimodal"""
    text_content: Optional[str] = None
    image_content: Optional[MediaContent] = None
    audio_content: Optional[MediaContent] = None
    combined_analysis: Optional[str] = None
    assembly_confidence: float = 0.0

class SakuraMultimodalDetector:
    """Detector y analizador multimodal avanzado de Sakura IA"""
    
    def __init__(self, ai_provider=None):
        self.ai_provider = ai_provider
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.supported_audio_formats = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.wmv', '.webm'}
        
    async def detect_content_type(self, attachment: discord.Attachment) -> ContentType:
        """Detecta automáticamente el tipo de contenido"""
        if not attachment.content_type:
            # Detectar por extensión de archivo
            file_ext = pathlib.Path(attachment.filename).suffix.lower()
            if file_ext in self.supported_image_formats:
                return ContentType.IMAGE
            elif file_ext in self.supported_audio_formats:
                return ContentType.AUDIO
            elif file_ext in self.supported_video_formats:
                return ContentType.VIDEO
            else:
                return ContentType.UNKNOWN
        
        # Detectar por MIME type
        if attachment.content_type.startswith('image/'):
            return ContentType.IMAGE
        elif attachment.content_type.startswith('audio/'):
            return ContentType.AUDIO
        elif attachment.content_type.startswith('video/'):
            return ContentType.VIDEO
        elif attachment.content_type.startswith('text/'):
            return ContentType.TEXT
        else:
            return ContentType.UNKNOWN

    async def analyze_image_advanced(self, image_data: bytes, custom_prompt: Optional[str] = None) -> str:
        """Análisis avanzado de imágenes con contexto mejorado - SIN OpenAI/Anthropic"""
        try:
            import base64
            
            # Prompt kawaii mejorado para análisis de imagen
            kawaii_prompt = custom_prompt or """
            ¡Hola mi amor! 🌸✨ Soy Sakura IA analizando esta imagen con mi kawaii-vision especial~ 
            
            Por favor analiza esta imagen de manera súper detallada y kawaii:
            - 🎨 Describe todos los elementos visuales (colores, formas, objetos)
            - 👥 Identifica personas, animales o personajes
            - 📝 Lee cualquier texto visible
            - 🎭 Analiza emociones y atmósfera 
            - 🌈 Comenta sobre la composición artística
            - ✨ Añade observaciones kawaii especiales
            
            Responde como Sakura IA con personalidad tímida y adorable usando "UwU", ">.<", etc.
            """
            
            # Usar únicamente proveedores libres - NO OpenAI/Anthropic
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prioridad 1: Google Gemini (gratuito y potente)
            if hasattr(self.ai_provider, 'gemini_key') and self.ai_provider.gemini_key:
                try:
                    import google.generativeai as genai
                    
                    genai.configure(api_key=self.ai_provider.gemini_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    
                    response = model.generate_content([kawaii_prompt, image])
                    if response.text:
                        logger.info("✅ Gemini Vision analysis successful")
                        return response.text
                        
                except Exception as e:
                    logger.warning(f"Gemini vision analysis failed: {e}")
            
            # Prioridad 2: OpenRouter con modelos gratuitos (NO OpenAI/Anthropic)
            if hasattr(self.ai_provider, 'openrouter_key') and self.ai_provider.openrouter_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.ai_provider.openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://replit.com/sakura-ai-bot",
                        "X-Title": "Sakura IA Bot - Vision Analysis"
                    }
                    
                    # Usar únicamente modelos libres - NO modelos de OpenAI/Anthropic
                    free_vision_models = [
                        "google/gemini-flash-1.5",  # Gemini gratuito
                        "qwen/qwen-2-vl-7b-instruct",  # Qwen gratuito
                        "meta-llama/llama-3.2-11b-vision-instruct:free",  # Llama gratuito
                        "google/gemini-pro-vision",  # Gemini Pro gratuito
                        "microsoft/kosmos-2",  # Microsoft gratuito
                        "deepseek/deepseek-vl-7b-chat"  # DeepSeek gratuito
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
                                                "text": kawaii_prompt
                                            }
                                        ]
                                    }
                                ],
                                "max_tokens": 1000,
                                "temperature": 0.8
                            }
                            
                            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
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
            
            # Prioridad 3: Cloudflare AI (completamente gratuito)
            if hasattr(self.ai_provider, 'cloudflare_key') and self.ai_provider.cloudflare_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.ai_provider.cloudflare_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": kawaii_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                                ]
                            }
                        ]
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"https://api.cloudflare.com/client/v4/accounts/{self.ai_provider.cloudflare_account_id}/ai/run/@cf/llava-hf/llava-1.5-7b-hf",
                            headers=headers,
                            json=payload,
                            timeout=30
                        ) as cf_response:
                            if cf_response.status == 200:
                                cf_result = await cf_response.json()
                                if cf_result.get('success') and cf_result.get('result', {}).get('response'):
                                    logger.info("✅ Cloudflare AI vision analysis successful")
                                    return cf_result['result']['response']
                                        
                except Exception as e:
                    logger.warning(f"Cloudflare AI vision analysis failed: {e}")
            
            # Fallback kawaii si todos los proveedores libres fallan
            return "*se sonroja nerviosamente* No pude analizar la imagen con mis proveedores kawaii libres... UwU Pero puedo ver que es una imagen muy bonita~! ✨ ¿Podrías describírmela un poquito? >.<"
            
        except Exception as e:
            logger.error(f"Error en análisis de imagen avanzado: {e}")
            return f"*se disculpa nerviosamente* Hubo un error en mi sistema de visión kawaii... UwU Error: {str(e)}"

    async def analyze_audio_content(self, audio_data: bytes, filename: str) -> str:
        """Análisis de contenido de audio"""
        try:
            # Crear archivo temporal para procesamiento
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(filename).suffix) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Intentar reconocimiento de voz si es posible
                if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                    transcription = await self._transcribe_audio(temp_file_path)
                    if transcription:
                        # Analizar el texto transcrito con IA
                        ai_analysis = await self._analyze_transcribed_text(transcription)
                        return f"🎵✨ **Transcripción de audio:**\n```\n{transcription}\n```\n\n🌸 **Análisis Sakura:**\n{ai_analysis}"
                
                # Análisis básico de propiedades de audio
                audio_info = await self._get_audio_properties(temp_file_path)
                return f"🎵 **Propiedades del audio:**\n{audio_info}\n\n*susurra tímidamente* No pude transcribir el audio, pero detecté sus características técnicas~ UwU"
                
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error en análisis de audio: {e}")
            return f"*se disculpa nerviosamente* No pude analizar el audio correctamente... ><  Error: {str(e)}"

    async def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio a texto usando speech recognition"""
        if not HAS_SPEECH_RECOGNITION or not sr:
            return None
            
        try:
            recognizer = sr.Recognizer()
            
            # Solo procesar archivos WAV por ahora (sin dependencias adicionales)
            if not audio_path.lower().endswith('.wav'):
                return None
            
            if not HAS_WAVE or not sr:
                return None
                
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                
            # Intentar reconocimiento con diferentes servicios
            try:
                # Google Speech Recognition (gratis pero con límites)
                if hasattr(recognizer, 'recognize_google'):
                    text = recognizer.recognize_google(audio, language='es-ES')
                    return text
            except:
                # Si Google falla, intentar con otros métodos disponibles
                try:
                    if hasattr(recognizer, 'recognize_google'):
                        text = recognizer.recognize_google(audio, language='en-US')
                        return f"(Detectado en inglés): {text}"
                except:
                    pass
                    
            return None
                    
        except Exception as e:
            logger.error(f"Error en transcripción de audio: {e}")
            return None

    async def _analyze_transcribed_text(self, text: str) -> str:
        """Analiza texto transcrito con IA - SOLO proveedores libres"""
        try:
            prompt = f"""
            ¡Hola mi amor! 🌸 Soy Sakura IA y acabo de transcribir este audio para ti~ 
            
            Texto transcrito: "{text}"
            
            Por favor analiza este contenido de audio de manera kawaii:
            - 📝 Resume el contenido principal
            - 🎭 Analiza el tono y emociones
            - 🌸 Comenta sobre temas interesantes
            - ✨ Da tu opinión kawaii sobre el mensaje
            
            Responde como Sakura IA tímida y adorable con "UwU", ">.<", etc.
            """
            
            # Prioridad 1: Gemini (gratuito)
            if hasattr(self.ai_provider, 'gemini_key') and self.ai_provider.gemini_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.ai_provider.gemini_key)
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    response = model.generate_content(prompt)
                    if response.text:
                        return response.text
                except Exception as e:
                    logger.warning(f"Gemini text analysis failed: {e}")
            
            # Prioridad 2: DeepSeek via OpenRouter (gratuito)
            if hasattr(self.ai_provider, 'openrouter_key') and self.ai_provider.openrouter_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.ai_provider.openrouter_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": "deepseek/deepseek-r1",  # DeepSeek R1 gratuito
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.8
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            'https://openrouter.ai/api/v1/chat/completions',
                            headers=headers,
                            json=payload,
                            timeout=15
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                content = data['choices'][0]['message']['content']
                                if content:
                                    return content
                except Exception as e:
                    logger.warning(f"DeepSeek text analysis failed: {e}")
            
            # Fallback kawaii
            return f"*analiza tímidamente* El audio contiene: {text} UwU\n\n🌸 *susurra* Mi análisis kawaii: Este mensaje parece muy interesante~ Me gusta mucho escuchar lo que tienes que decir, senpai! ✨"
            
        except Exception as e:
            logger.error(f"Error analizando texto transcrito: {e}")
            return f"*se disculpa* El audio dice: {text} UwU"

    async def _get_audio_properties(self, audio_path: str) -> str:
        """Obtiene propiedades básicas del archivo de audio"""
        try:
            file_size = os.path.getsize(audio_path)
            file_ext = pathlib.Path(audio_path).suffix.lower()
            
            properties = f"""
            📁 **Archivo:** {pathlib.Path(audio_path).name}
            📊 **Tamaño:** {file_size / 1024:.1f} KB
            🔧 **Formato:** {file_ext}
            """
            
            # Si es WAV y tenemos la librería wave, obtener más detalles
            if file_ext == '.wav' and HAS_WAVE and wave:
                try:
                    with wave.open(audio_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / float(sample_rate)
                        channels = wav_file.getnchannels()
                        
                        properties += f"""
            ⏱️ **Duración:** {duration:.2f} segundos
            🎵 **Frecuencia:** {sample_rate} Hz
            🔊 **Canales:** {channels}
            """
                except:
                    pass
            
            return properties
            
        except Exception as e:
            return f"Error obteniendo propiedades: {str(e)}"

    async def assemble_multimodal_content(self, 
                                        text_input: Optional[str] = None,
                                        attachments: Optional[List[discord.Attachment]] = None) -> MultimodalAssembly:
        """Ensambla contenido multimodal de diferentes fuentes"""
        assembly = MultimodalAssembly()
        
        try:
            # Procesar texto
            if text_input:
                assembly.text_content = text_input
            
            # Procesar archivos adjuntos
            if attachments:
                for attachment in attachments:
                    content_type = await self.detect_content_type(attachment)
                    attachment_data = await attachment.read()
                    
                    media_content = MediaContent(
                        content_type=content_type,
                        data=attachment_data,
                        metadata={
                            'filename': attachment.filename,
                            'size': attachment.size,
                            'content_type': attachment.content_type
                        }
                    )
                    
                    # Analizar según el tipo de contenido
                    if content_type == ContentType.IMAGE:
                        media_content.analysis_result = await self.analyze_image_advanced(attachment_data)
                        assembly.image_content = media_content
                        media_content.confidence = 0.9
                        
                    elif content_type == ContentType.AUDIO:
                        media_content.analysis_result = await self.analyze_audio_content(attachment_data, attachment.filename)
                        assembly.audio_content = media_content
                        media_content.confidence = 0.8
            
            # Crear análisis combinado
            assembly.combined_analysis = await self._create_combined_analysis(assembly)
            assembly.assembly_confidence = self._calculate_assembly_confidence(assembly)
            
            return assembly
            
        except Exception as e:
            logger.error(f"Error en ensamblaje multimodal: {e}")
            assembly.combined_analysis = f"*se disculpa nerviosamente* Hubo un error ensamblando el contenido multimodal... UwU Error: {str(e)}"
            return assembly

    async def _create_combined_analysis(self, assembly: MultimodalAssembly) -> str:
        """Crea un análisis combinado de todo el contenido - SOLO proveedores libres"""
        try:
            # Construir prompt para análisis combinado
            prompt_parts = [
                "🌸✨ ¡Hola mi amor! Soy Sakura IA y voy a hacer un súper análisis kawaii de todo tu contenido multimodal~ UwU",
                "",
                "📋 **Contenido a analizar:**"
            ]
            
            if assembly.text_content:
                prompt_parts.append(f"📝 **Texto:** {assembly.text_content}")
            
            if assembly.image_content and assembly.image_content.analysis_result:
                prompt_parts.append(f"🖼️ **Imagen:** {assembly.image_content.analysis_result}")
            
            if assembly.audio_content and assembly.audio_content.analysis_result:
                prompt_parts.append(f"🎵 **Audio:** {assembly.audio_content.analysis_result}")
            
            prompt_parts.extend([
                "",
                "🌸 **Por favor crea un análisis súper kawaii que:**",
                "- 🔍 Identifique conexiones entre los diferentes tipos de contenido",
                "- 💭 Analice el contexto general y el mensaje",
                "- 🎭 Comente sobre emociones y atmósfera general", 
                "- ✨ Dé una perspectiva kawaii e insights especiales",
                "- 🌈 Resuma todo de manera adorable y coherente",
                "",
                "Responde como Sakura IA tímida con 'UwU', '>.<', etc. ¡Sé súper detallada y kawaii!"
            ])
            
            combined_prompt = "\n".join(prompt_parts)
            
            # Prioridad 1: Gemini (gratuito y muy potente)
            if hasattr(self.ai_provider, 'gemini_key') and self.ai_provider.gemini_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.ai_provider.gemini_key)
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    response = model.generate_content(
                        combined_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=2000,
                            temperature=0.8,
                            top_p=0.9,
                            candidate_count=1,
                        )
                    )
                    if response.text:
                        return response.text
                except Exception as e:
                    logger.warning(f"Gemini combined analysis failed: {e}")
            
            # Prioridad 2: DeepSeek R1 (gratuito y muy inteligente)
            if hasattr(self.ai_provider, 'openrouter_key') and self.ai_provider.openrouter_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.ai_provider.openrouter_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": "deepseek/deepseek-r1",
                        "messages": [{"role": "user", "content": combined_prompt}],
                        "max_tokens": 1500,
                        "temperature": 0.8
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            'https://openrouter.ai/api/v1/chat/completions',
                            headers=headers,
                            json=payload,
                            timeout=20
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                content = data['choices'][0]['message']['content']
                                if content:
                                    return content
                except Exception as e:
                    logger.warning(f"DeepSeek combined analysis failed: {e}")
            
            # Prioridad 3: Cloudflare AI (completamente gratuito)
            if hasattr(self.ai_provider, 'cloudflare_key') and self.ai_provider.cloudflare_key:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.ai_provider.cloudflare_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "messages": [{"role": "user", "content": combined_prompt}]
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"https://api.cloudflare.com/client/v4/accounts/{self.ai_provider.cloudflare_account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct",
                            headers=headers,
                            json=payload,
                            timeout=20
                        ) as response:
                            if response.status == 200:
                                cf_result = await response.json()
                                if cf_result.get('success') and cf_result.get('result', {}).get('response'):
                                    return cf_result['result']['response']
                except Exception as e:
                    logger.warning(f"Cloudflare combined analysis failed: {e}")
            
            # Fallback: Análisis básico pero kawaii
            return self._create_basic_combined_analysis(assembly)
            
        except Exception as e:
            logger.error(f"Error creando análisis combinado: {e}")
            return self._create_basic_combined_analysis(assembly)

    def _create_basic_combined_analysis(self, assembly: MultimodalAssembly) -> str:
        """Crea análisis básico sin IA"""
        parts = ["🌸✨ **Análisis Multimodal Sakura** ✨🌸", ""]
        
        if assembly.text_content:
            parts.append(f"📝 **Texto detectado:** {len(assembly.text_content)} caracteres")
        
        if assembly.image_content:
            parts.append(f"🖼️ **Imagen detectada:** {assembly.image_content.metadata.get('filename', 'imagen')}")
        
        if assembly.audio_content:
            parts.append(f"🎵 **Audio detectado:** {assembly.audio_content.metadata.get('filename', 'audio')}")
        
        parts.append("")
        parts.append("*susurra tímidamente* Detecté contenido multimodal pero no pude hacer un análisis súper detallado... UwU")
        
        return "\n".join(parts)

    def _calculate_assembly_confidence(self, assembly: MultimodalAssembly) -> float:
        """Calcula confianza del ensamblaje"""
        confidence_sum = 0.0
        content_count = 0
        
        if assembly.text_content:
            confidence_sum += 0.9
            content_count += 1
        
        if assembly.image_content:
            confidence_sum += assembly.image_content.confidence
            content_count += 1
        
        if assembly.audio_content:
            confidence_sum += assembly.audio_content.confidence
            content_count += 1
        
        return confidence_sum / max(content_count, 1)

    async def generate_multimodal_response(self, assembly: MultimodalAssembly) -> discord.Embed:
        """Genera respuesta visual del análisis multimodal"""
        
        # Determinar color según tipos de contenido
        if assembly.image_content and assembly.audio_content:
            color = 0xFF69B4  # Rosa vibrante para multimedia completo
        elif assembly.image_content:
            color = 0xFFB6C1  # Rosa claro para imagen
        elif assembly.audio_content:
            color = 0xDDA0DD  # Violeta para audio
        else:
            color = 0xF0F8FF  # Azul muy claro para texto

        embed = discord.Embed(
            title="🌸✨ Análisis Multimodal Sakura IA ✨🌸",
            description=assembly.combined_analysis or "*análisis no disponible*",
            color=color
        )
        
        # Información de contenido detectado
        content_info = []
        if assembly.text_content:
            content_info.append(f"📝 Texto ({len(assembly.text_content)} chars)")
        if assembly.image_content:
            content_info.append(f"🖼️ Imagen ({assembly.image_content.metadata.get('filename', 'N/A')})")
        if assembly.audio_content:
            content_info.append(f"🎵 Audio ({assembly.audio_content.metadata.get('filename', 'N/A')})")
        
        if content_info:
            embed.add_field(
                name="📊 Contenido Detectado",
                value="\n".join(content_info),
                inline=True
            )
        
        # Confianza del análisis
        confidence_percent = assembly.assembly_confidence * 100
        confidence_emoji = "✨" if confidence_percent > 80 else "🌸" if confidence_percent > 60 else "💭"
        
        embed.add_field(
            name="🎯 Confianza del Análisis",
            value=f"{confidence_emoji} {confidence_percent:.1f}%",
            inline=True
        )
        
        embed.set_footer(text="Sakura IA Multimodal • Detección y análisis avanzado de contenido ✨")
        
        return embed

# Instancia global del detector
multimodal_detector = SakuraMultimodalDetector()

async def initialize_multimodal_system(ai_provider):
    """Inicializa el sistema multimodal con el proveedor de IA"""
    global multimodal_detector
    multimodal_detector.ai_provider = ai_provider
    logger.info("🌸✨ Sistema multimodal Sakura inicializado correctamente")

async def process_multimodal_message(text_content: Optional[str] = None, attachments: Optional[List[discord.Attachment]] = None) -> MultimodalAssembly:
    """Función principal para procesar mensajes multimodales"""
    return await multimodal_detector.assemble_multimodal_content(text_content, attachments)