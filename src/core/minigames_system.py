"""
🎮 Sistema de Mini-Juegos Interactivos para Sakura Bot
=====================================================
Juegos kawaii y divertidos con sistema de recompensas
"""

import discord
from discord.ext import commands
from discord import app_commands
import random
import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class MiniGamesSystem:
    """Sistema de mini-juegos con recompensas"""
    
    def __init__(self, bot):
        self.bot = bot
        self.active_games = {}  # user_id: game_data
        self.daily_rewards = {}  # user_id: last_claim_date
        
        # Configuración de juegos
        self.games_config = {
            "trivia": {
                "reward_base": 50,
                "questions": [
                    {"q": "¿Cuál es el color favorito de Sakura?", "a": ["rosa", "pink"], "hint": "Es un color muy kawaii 🌸"},
                    {"q": "¿Qué animal representa la ternura?", "a": ["gato", "cat", "gatito"], "hint": "Hace 'miau' 🐱"},
                    {"q": "¿Cuántos pétalos tiene una flor de cerezo típica?", "a": ["5", "cinco"], "hint": "Los dedos de una mano ✋"},
                    {"q": "¿En qué país se originó el anime?", "a": ["japon", "japan"], "hint": "Tierra del sol naciente 🇯🇵"},
                    {"q": "¿Qué significa 'kawaii' en japonés?", "a": ["lindo", "cute", "adorable"], "hint": "Algo muy tierno 💖"}
                ]
            },
            "memory": {
                "reward_base": 75,
                "emojis": ["🌸", "💖", "✨", "🦄", "🌙", "⭐", "🍰", "🎀"]
            },
            "reaction": {
                "reward_base": 30,
                "emojis": ["🚀", "⚡", "💨", "🔥", "⭐"]
            }
        }
    
    async def start_trivia_game(self, interaction: discord.Interaction):
        """Iniciar juego de trivia kawaii"""
        user_id = interaction.user.id
        
        if user_id in self.active_games:
            await interaction.response.send_message("¡Ya tienes un juego activo! 🎮", ephemeral=True)
            return
        
        question_data = random.choice(self.games_config["trivia"]["questions"])
        
        embed = discord.Embed(
            title="🧠 Sakura Trivia Kawaii",
            description=f"**Pregunta:** {question_data['q']}",
            color=0xFF69B4
        )
        embed.add_field(name="💡 Pista", value=question_data['hint'], inline=False)
        embed.add_field(name="⏰ Tiempo", value="60 segundos para responder", inline=False)
        embed.set_footer(text="Escribe tu respuesta en el chat!")
        
        await interaction.response.send_message(embed=embed)
        
        # Guardar datos del juego
        self.active_games[user_id] = {
            "type": "trivia",
            "data": question_data,
            "start_time": datetime.now(),
            "channel_id": interaction.channel.id
        }
        
        # Timeout del juego
        await asyncio.sleep(60)
        if user_id in self.active_games and self.active_games[user_id]["type"] == "trivia":
            timeout_embed = discord.Embed(
                title="⏰ ¡Tiempo Agotado!",
                description=f"La respuesta era: **{question_data['a'][0]}**",
                color=0xFF0000
            )
            await interaction.followup.send(embed=timeout_embed)
            del self.active_games[user_id]
    
    async def start_memory_game(self, interaction: discord.Interaction, difficulty: str = "normal"):
        """Iniciar juego de memoria"""
        user_id = interaction.user.id
        
        if user_id in self.active_games:
            await interaction.response.send_message("¡Ya tienes un juego activo! 🎮", ephemeral=True)
            return
        
        # Configurar dificultad
        difficulty_config = {
            "facil": {"pairs": 3, "time": 15, "multiplier": 1},
            "normal": {"pairs": 4, "time": 12, "multiplier": 1.5},
            "dificil": {"pairs": 6, "time": 8, "multiplier": 2}
        }
        
        config = difficulty_config.get(difficulty, difficulty_config["normal"])
        emojis = random.sample(self.games_config["memory"]["emojis"], config["pairs"])
        sequence = emojis * 2
        random.shuffle(sequence)
        
        # Mostrar secuencia brevemente
        embed = discord.Embed(
            title="🧠 Juego de Memoria Kawaii",
            description="¡Memoriza la secuencia! 💖",
            color=0xFF69B4
        )
        embed.add_field(name="🎯 Secuencia", value=" ".join(sequence), inline=False)
        embed.add_field(name="⏰ Tiempo para memorizar", value=f"{config['time']} segundos", inline=False)
        
        await interaction.response.send_message(embed=embed)
        
        # Esperar tiempo de memorización
        await asyncio.sleep(config["time"])
        
        # Ocultar secuencia y pedir respuesta
        hidden_embed = discord.Embed(
            title="🧠 ¡Ahora escribe la secuencia!",
            description="Escribe los emojis en el orden correcto (separados por espacios)",
            color=0xFFD700
        )
        
        await interaction.edit_original_response(embed=hidden_embed)
        
        # Guardar datos del juego
        self.active_games[user_id] = {
            "type": "memory",
            "data": {"sequence": sequence, "difficulty": difficulty},
            "start_time": datetime.now(),
            "channel_id": interaction.channel.id
        }
    
    async def start_reaction_game(self, interaction: discord.Interaction):
        """Iniciar juego de velocidad de reacción"""
        user_id = interaction.user.id
        
        if user_id in self.active_games:
            await interaction.response.send_message("¡Ya tienes un juego activo! 🎮", ephemeral=True)
            return
        
        embed = discord.Embed(
            title="⚡ Juego de Reacción Rápida",
            description="¡Espera a que aparezca el emoji especial y reacciona lo más rápido posible!",
            color=0xFF69B4
        )
        
        await interaction.response.send_message(embed=embed)
        
        # Esperar tiempo aleatorio
        wait_time = random.randint(3, 8)
        await asyncio.sleep(wait_time)
        
        # Mostrar emoji objetivo
        target_emoji = random.choice(self.games_config["reaction"]["emojis"])
        
        reaction_embed = discord.Embed(
            title="⚡ ¡AHORA!",
            description=f"¡Reacciona con {target_emoji} lo más rápido posible!",
            color=0x00FF00
        )
        
        message = await interaction.edit_original_response(embed=reaction_embed)
        await message.add_reaction(target_emoji)
        
        # Guardar datos del juego
        self.active_games[user_id] = {
            "type": "reaction",
            "data": {"emoji": target_emoji, "message_id": message.id},
            "start_time": datetime.now(),
            "channel_id": interaction.channel.id
        }
    
    async def check_trivia_answer(self, message):
        """Verificar respuesta de trivia"""
        user_id = message.author.id
        
        if user_id not in self.active_games or self.active_games[user_id]["type"] != "trivia":
            return
        
        game_data = self.active_games[user_id]
        if message.channel.id != game_data["channel_id"]:
            return
        
        user_answer = message.content.lower().strip()
        correct_answers = [ans.lower() for ans in game_data["data"]["a"]]
        
        if any(ans in user_answer for ans in correct_answers):
            # Respuesta correcta
            reward = self.games_config["trivia"]["reward_base"]
            
            embed = discord.Embed(
                title="🎉 ¡Correcto!",
                description=f"¡Excelente! Has ganado **{reward}** monedas kawaii 🌸",
                color=0x00FF00
            )
            embed.set_footer(text=f"Tiempo: {(datetime.now() - game_data['start_time']).seconds}s")
            
            await message.reply(embed=embed)
            
            # Dar recompensa (aquí se conectaría con el sistema de economía)
            await self.give_reward(user_id, reward)
            
        else:
            # Respuesta incorrecta
            embed = discord.Embed(
                title="❌ Incorrecto",
                description=f"La respuesta correcta era: **{game_data['data']['a'][0]}**",
                color=0xFF0000
            )
            await message.reply(embed=embed)
        
        # Limpiar juego
        del self.active_games[user_id]
    
    async def check_memory_answer(self, message):
        """Verificar respuesta de memoria"""
        user_id = message.author.id
        
        if user_id not in self.active_games or self.active_games[user_id]["type"] != "memory":
            return
        
        game_data = self.active_games[user_id]
        if message.channel.id != game_data["channel_id"]:
            return
        
        user_sequence = message.content.split()
        correct_sequence = game_data["data"]["sequence"]
        
        if user_sequence == correct_sequence:
            # Respuesta correcta
            difficulty = game_data["data"]["difficulty"]
            multiplier = {"facil": 1, "normal": 1.5, "dificil": 2}.get(difficulty, 1)
            reward = int(self.games_config["memory"]["reward_base"] * multiplier)
            
            embed = discord.Embed(
                title="🧠 ¡Memoria Perfecta!",
                description=f"¡Increíble memoria! Has ganado **{reward}** monedas kawaii 🌸",
                color=0x00FF00
            )
            
            await message.reply(embed=embed)
            await self.give_reward(user_id, reward)
            
        else:
            embed = discord.Embed(
                title="❌ Secuencia Incorrecta",
                description=f"La secuencia correcta era: {' '.join(correct_sequence)}",
                color=0xFF0000
            )
            await message.reply(embed=embed)
        
        del self.active_games[user_id]
    
    async def give_reward(self, user_id: int, amount: int):
        """Dar recompensa al usuario (conectar con sistema de economía)"""
        # Aquí se conectaría con PostgreSQL para actualizar el balance
        try:
            # Ejemplo de conexión con la base de datos
            # await bot.postgresql_manager.update_user_balance(user_id, amount)
            pass
        except Exception as e:
            print(f"Error dando recompensa: {e}")

class MiniGamesCommands(commands.Cog):
    """Comandos de mini-juegos"""
    
    def __init__(self, bot):
        self.bot = bot
        self.games_system = MiniGamesSystem(bot)
    
    @app_commands.command(name="trivia", description="🧠 Jugar trivia kawaii y ganar monedas")
    async def play_trivia(self, interaction: discord.Interaction):
        """Jugar trivia"""
        await self.games_system.start_trivia_game(interaction)
    
    @app_commands.command(name="memoria", description="🧠 Juego de memoria con emojis kawaii")
    @app_commands.describe(dificultad="Nivel de dificultad (facil, normal, dificil)")
    async def play_memory(self, interaction: discord.Interaction, dificultad: str = "normal"):
        """Jugar memoria"""
        await self.games_system.start_memory_game(interaction, dificultad)
    
    @app_commands.command(name="reaccion", description="⚡ Juego de velocidad de reacciones")
    async def play_reaction(self, interaction: discord.Interaction):
        """Jugar reacción rápida"""
        await self.games_system.start_reaction_game(interaction)
    
    @app_commands.command(name="juegos_diarios", description="🎁 Reclamar recompensa diaria de juegos")
    async def daily_games_reward(self, interaction: discord.Interaction):
        """Recompensa diaria por jugar"""
        user_id = interaction.user.id
        today = datetime.now().date()
        
        if user_id in self.games_system.daily_rewards:
            last_claim = self.games_system.daily_rewards[user_id]
            if last_claim >= today:
                embed = discord.Embed(
                    title="⏰ Ya Reclamado",
                    description="¡Ya reclamaste tu recompensa diaria! Vuelve mañana 🌸",
                    color=0xFFD700
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
        
        # Dar recompensa diaria
        daily_reward = 100
        self.games_system.daily_rewards[user_id] = today
        
        embed = discord.Embed(
            title="🎁 ¡Recompensa Diaria!",
            description=f"¡Has recibido **{daily_reward}** monedas kawaii por ser un jugador activo! 🌸✨",
            color=0x00FF7F
        )
        embed.add_field(name="🎮 Mini-juegos disponibles", value="`/trivia` • `/memoria` • `/reaccion`", inline=False)
        
        await interaction.response.send_message(embed=embed)
        await self.games_system.give_reward(user_id, daily_reward)
    
    @commands.Cog.listener()
    async def on_message(self, message):
        """Escuchar respuestas de juegos"""
        if message.author.bot:
            return
        
        # Verificar respuestas de trivia y memoria
        await self.games_system.check_trivia_answer(message)
        await self.games_system.check_memory_answer(message)
    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Manejar reacciones de juegos"""
        if user.bot:
            return
        
        user_id = user.id
        
        if user_id in self.games_system.active_games:
            game_data = self.games_system.active_games[user_id]
            
            if (game_data["type"] == "reaction" and 
                reaction.message.id == game_data["data"]["message_id"] and
                str(reaction.emoji) == game_data["data"]["emoji"]):
                
                # Calcular tiempo de reacción
                reaction_time = (datetime.now() - game_data["start_time"]).total_seconds()
                
                # Calcular recompensa basada en velocidad
                base_reward = self.games_system.games_config["reaction"]["reward_base"]
                if reaction_time < 1:
                    reward = base_reward * 3  # Super rápido
                elif reaction_time < 2:
                    reward = base_reward * 2  # Rápido
                else:
                    reward = base_reward  # Normal
                
                embed = discord.Embed(
                    title="⚡ ¡Reacción Rápida!",
                    description=f"¡Tiempo: {reaction_time:.2f}s!\nHas ganado **{reward}** monedas kawaii 🌸",
                    color=0x00FF00
                )
                
                await reaction.message.reply(embed=embed)
                await self.games_system.give_reward(user_id, reward)
                
                del self.games_system.active_games[user_id]

async def setup(bot):
    await bot.add_cog(MiniGamesCommands(bot))