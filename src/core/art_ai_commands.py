"""
🎨 Sistema de Arte AI Avanzado para Sakura Bot
============================================
Genera arte personalizado usando múltiples modelos de IA
"""

import os
import aiohttp
import asyncio
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import discord
from discord.ext import commands
from discord import app_commands
import random

class ArtAISystem:
    """Sistema avanzado de generación de arte AI"""
    
    def __init__(self, bot):
        self.bot = bot
        self.huggingface_key = os.getenv('HUGGINGFACE_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
        # Estilos artísticos predefinidos
        self.art_styles = {
            "anime": "anime style, high quality, detailed, vibrant colors",
            "kawaii": "kawaii style, cute, pastel colors, adorable, chibi",
            "realistic": "photorealistic, high detail, professional photography",
            "fantasy": "fantasy art, magical, ethereal, mystical atmosphere",
            "cyberpunk": "cyberpunk style, neon lights, futuristic, digital art",
            "watercolor": "watercolor painting, soft colors, artistic brush strokes",
            "pixel": "pixel art style, retro gaming, 8-bit aesthetic",
            "manga": "manga style, black and white, dramatic shading"
        }
        
        # Prompts de ejemplo kawaii
        self.kawaii_prompts = [
            "cute magical girl with pink hair and sparkles",
            "adorable cat wearing a tiny crown",
            "kawaii food characters with happy faces",
            "magical forest with cute creatures",
            "pastel rainbow with smiling clouds"
        ]
    
    async def generate_dall_e_image(self, prompt: str, style: str = "anime") -> BytesIO:
        """Generar imagen usando DALL-E 3"""
        try:
            if not self.openai_key:
                return None
                
            styled_prompt = f"{prompt}, {self.art_styles.get(style, '')}"
            
            import openai
            client = openai.OpenAI(api_key=self.openai_key)
            
            response = await asyncio.to_thread(
                lambda: client.images.generate(
                    model="dall-e-3",
                    prompt=styled_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
            )
            
            image_url = response.data[0].url
            
            # Descargar la imagen
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        return BytesIO(image_data)
            
            return None
        except Exception as e:
            print(f"Error generando imagen DALL-E: {e}")
            return None
    
    async def generate_stable_diffusion_image(self, prompt: str, style: str = "anime") -> BytesIO:
        """Generar imagen usando Stable Diffusion XL"""
        try:
            if not self.huggingface_key:
                return None
                
            styled_prompt = f"{prompt}, {self.art_styles.get(style, '')}"
            
            headers = {
                'Authorization': f'Bearer {self.huggingface_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'inputs': styled_prompt,
                'parameters': {
                    'guidance_scale': 7.5,
                    'num_inference_steps': 30,
                    'width': 1024,
                    'height': 1024,
                    'negative_prompt': 'blurry, low quality, distorted, ugly'
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0',
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        return BytesIO(image_bytes)
            
            return None
        except Exception as e:
            print(f"Error generando imagen Stable Diffusion: {e}")
            return None
    
    def apply_kawaii_filter(self, image_bytes: BytesIO) -> BytesIO:
        """Aplicar filtros kawaii a una imagen"""
        try:
            # Cargar imagen
            image = Image.open(image_bytes)
            
            # Aumentar saturación para colores más vibrantes
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            
            # Aumentar brillo ligeramente
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # Aplicar un filtro suave
            image = image.filter(ImageFilter.SMOOTH)
            
            # Convertir de vuelta a bytes
            output = BytesIO()
            image.save(output, format='PNG')
            output.seek(0)
            
            return output
        except Exception as e:
            print(f"Error aplicando filtro kawaii: {e}")
            return image_bytes

class ArtCommands(commands.Cog):
    """Comandos de arte AI"""
    
    def __init__(self, bot):
        self.bot = bot
        self.art_system = ArtAISystem(bot)
    
    @app_commands.command(name="arte", description="🎨 Generar arte AI personalizado")
    @app_commands.describe(
        prompt="Descripción de lo que quieres crear",
        estilo="Estilo artístico (anime, kawaii, realistic, fantasy, etc.)"
    )
    async def generate_art(self, interaction: discord.Interaction, prompt: str, estilo: str = "anime"):
        """Generar arte usando IA"""
        await interaction.response.defer(thinking=True)
        
        try:
            # Crear embed de progreso
            embed = discord.Embed(
                title="🎨 Generando Arte AI",
                description=f"✨ Creando: `{prompt}`\n🎭 Estilo: `{estilo}`",
                color=0xFF69B4
            )
            embed.add_field(name="⏳ Progreso", value="Iniciando generación...", inline=False)
            
            await interaction.edit_original_response(embed=embed)
            
            # Intentar múltiples métodos
            image_data = None
            
            # 1. Intentar DALL-E 3 primero
            embed.set_field_at(0, name="⏳ Progreso", value="🤖 Usando DALL-E 3...", inline=False)
            await interaction.edit_original_response(embed=embed)
            
            image_data = await self.art_system.generate_dall_e_image(prompt, estilo)
            
            # 2. Si falla, intentar Stable Diffusion
            if not image_data:
                embed.set_field_at(0, name="⏳ Progreso", value="🎨 Usando Stable Diffusion XL...", inline=False)
                await interaction.edit_original_response(embed=embed)
                
                image_data = await self.art_system.generate_stable_diffusion_image(prompt, estilo)
            
            if image_data:
                # Aplicar filtros kawaii si es necesario
                if estilo == "kawaii":
                    embed.set_field_at(0, name="⏳ Progreso", value="🌸 Aplicando filtros kawaii...", inline=False)
                    await interaction.edit_original_response(embed=embed)
                    image_data = self.art_system.apply_kawaii_filter(image_data)
                
                # Crear embed final
                final_embed = discord.Embed(
                    title="🎨 ¡Arte AI Generado!",
                    description=f"✨ **Prompt:** {prompt}\n🎭 **Estilo:** {estilo.title()}",
                    color=0x00FF7F
                )
                final_embed.set_footer(text=f"Generado por {interaction.user.display_name} • Sakura IA Art")
                
                # Enviar imagen
                file = discord.File(image_data, filename=f"sakura_art_{interaction.user.id}.png")
                final_embed.set_image(url=f"attachment://sakura_art_{interaction.user.id}.png")
                
                await interaction.edit_original_response(embed=final_embed, attachments=[file])
                
            else:
                # Error embed
                error_embed = discord.Embed(
                    title="❌ Error Generando Arte",
                    description="No pude generar la imagen. Intenta con otro prompt o estilo.",
                    color=0xFF0000
                )
                await interaction.edit_original_response(embed=error_embed)
                
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Error",
                description=f"Ocurrió un error: {str(e)}",
                color=0xFF0000
            )
            await interaction.edit_original_response(embed=error_embed)
    
    @app_commands.command(name="arte_kawaii", description="🌸 Generar arte kawaii súper adorable")
    async def generate_kawaii_art(self, interaction: discord.Interaction):
        """Generar arte kawaii aleatorio"""
        random_prompt = random.choice(self.art_system.kawaii_prompts)
        await self.generate_art(interaction, random_prompt, "kawaii")
    
    @app_commands.command(name="estilos_arte", description="📋 Ver todos los estilos de arte disponibles")
    async def list_art_styles(self, interaction: discord.Interaction):
        """Mostrar estilos disponibles"""
        embed = discord.Embed(
            title="🎨 Estilos de Arte Disponibles",
            color=0xFF69B4
        )
        
        for style, description in self.art_system.art_styles.items():
            embed.add_field(
                name=f"🎭 {style.title()}",
                value=description,
                inline=False
            )
        
        embed.set_footer(text="Usa /arte [prompt] [estilo] para generar arte personalizado")
        await interaction.response.send_message(embed=embed)

async def setup(bot):
    await bot.add_cog(ArtCommands(bot))