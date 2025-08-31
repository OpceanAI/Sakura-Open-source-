"""
🏆 Sistema de Niveles y Logros para Sakura Bot
============================================
Sistema completo de progresión con logros kawaii
"""

import discord
from discord.ext import commands
from discord import app_commands
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class Achievement:
    """Clase para representar un logro"""
    id: str
    name: str
    description: str
    emoji: str
    rarity: str  # common, rare, epic, legendary
    reward: int
    requirement: dict

class AchievementsSystem:
    """Sistema de logros y niveles"""
    
    def __init__(self, bot):
        self.bot = bot
        
        # Configuración de niveles
        self.level_config = {
            "xp_base": 100,
            "xp_multiplier": 1.5,
            "rewards_per_level": 50
        }
        
        # Definir logros
        self.achievements = {
            # Logros de chat
            "chatter": Achievement(
                "chatter", "Charlador Kawaii", "Envía 100 mensajes", 
                "💬", "common", 100, {"type": "messages", "count": 100}
            ),
            "social_butterfly": Achievement(
                "social_butterfly", "Mariposa Social", "Envía 1000 mensajes", 
                "🦋", "rare", 500, {"type": "messages", "count": 1000}
            ),
            
            # Logros de música
            "music_lover": Achievement(
                "music_lover", "Amante de la Música", "Reproduce 50 canciones", 
                "🎵", "common", 150, {"type": "songs_played", "count": 50}
            ),
            "dj_master": Achievement(
                "dj_master", "DJ Master", "Reproduce 500 canciones", 
                "🎧", "epic", 1000, {"type": "songs_played", "count": 500}
            ),
            
            # Logros de juegos
            "gamer": Achievement(
                "gamer", "Jugador Kawaii", "Juega 10 mini-juegos", 
                "🎮", "common", 200, {"type": "games_played", "count": 10}
            ),
            "trivia_master": Achievement(
                "trivia_master", "Maestro del Trivia", "Responde 25 preguntas correctamente", 
                "🧠", "rare", 750, {"type": "trivia_correct", "count": 25}
            ),
            
            # Logros de tiempo
            "early_bird": Achievement(
                "early_bird", "Madrugador", "Usa el bot antes de las 6 AM", 
                "🌅", "rare", 300, {"type": "time_based", "hour": 6}
            ),
            "night_owl": Achievement(
                "night_owl", "Búho Nocturno", "Usa el bot después de las 11 PM", 
                "🦉", "rare", 300, {"type": "time_based", "hour": 23}
            ),
            
            # Logros especiales
            "first_steps": Achievement(
                "first_steps", "Primeros Pasos", "Usa el bot por primera vez", 
                "👶", "common", 50, {"type": "first_use", "count": 1}
            ),
            "loyal_user": Achievement(
                "loyal_user", "Usuario Leal", "Usa el bot 30 días consecutivos", 
                "👑", "legendary", 2000, {"type": "consecutive_days", "count": 30}
            ),
            "sakura_fan": Achievement(
                "sakura_fan", "Fan de Sakura", "Interactúa con IA 100 veces", 
                "🌸", "epic", 800, {"type": "ai_interactions", "count": 100}
            )
        }
        
        # Colores por rareza
        self.rarity_colors = {
            "common": 0x808080,    # Gris
            "rare": 0x0080FF,      # Azul
            "epic": 0x8000FF,      # Púrpura
            "legendary": 0xFFD700   # Dorado
        }
    
    def calculate_level_xp(self, level: int) -> int:
        """Calcular XP necesaria para un nivel"""
        return int(self.level_config["xp_base"] * (self.level_config["xp_multiplier"] ** (level - 1)))
    
    def get_level_from_xp(self, total_xp: int) -> tuple:
        """Obtener nivel y progreso desde XP total"""
        level = 1
        xp_for_current = 0
        
        while True:
            xp_needed = self.calculate_level_xp(level)
            if total_xp < xp_for_current + xp_needed:
                break
            xp_for_current += xp_needed
            level += 1
        
        current_level_xp = total_xp - xp_for_current
        next_level_xp = self.calculate_level_xp(level)
        
        return level, current_level_xp, next_level_xp
    
    async def add_xp(self, user_id: int, amount: int) -> dict:
        """Agregar XP y verificar subida de nivel"""
        # Aquí se conectaría con PostgreSQL
        # Por ahora simulamos los datos
        
        old_xp = 0  # Obtener de DB
        new_xp = old_xp + amount
        
        old_level, _, _ = self.get_level_from_xp(old_xp)
        new_level, current_xp, next_xp = self.get_level_from_xp(new_xp)
        
        result = {
            "old_level": old_level,
            "new_level": new_level,
            "current_xp": current_xp,
            "next_level_xp": next_xp,
            "total_xp": new_xp,
            "xp_gained": amount,
            "level_up": new_level > old_level
        }
        
        return result
    
    async def check_achievements(self, user_id: int, action_type: str, **kwargs) -> List[Achievement]:
        """Verificar si se desbloquearon logros"""
        # Obtener estadísticas del usuario de la DB
        user_stats = await self.get_user_stats(user_id)
        unlocked_achievements = []
        
        for achievement in self.achievements.values():
            if await self.is_achievement_unlocked(user_id, achievement.id):
                continue  # Ya desbloqueado
            
            if await self.check_achievement_requirement(user_stats, achievement, action_type, **kwargs):
                unlocked_achievements.append(achievement)
                await self.unlock_achievement(user_id, achievement.id)
        
        return unlocked_achievements
    
    async def check_achievement_requirement(self, user_stats: dict, achievement: Achievement, action_type: str, **kwargs) -> bool:
        """Verificar si se cumple el requisito de un logro"""
        req = achievement.requirement
        
        if req["type"] != action_type:
            return False
        
        if req["type"] == "messages":
            return user_stats.get("messages_sent", 0) >= req["count"]
        elif req["type"] == "songs_played":
            return user_stats.get("songs_played", 0) >= req["count"]
        elif req["type"] == "games_played":
            return user_stats.get("games_played", 0) >= req["count"]
        elif req["type"] == "trivia_correct":
            return user_stats.get("trivia_correct", 0) >= req["count"]
        elif req["type"] == "time_based":
            current_hour = datetime.now().hour
            if achievement.id == "early_bird":
                return current_hour < req["hour"]
            elif achievement.id == "night_owl":
                return current_hour >= req["hour"]
        elif req["type"] == "first_use":
            return user_stats.get("first_use", False)
        elif req["type"] == "ai_interactions":
            return user_stats.get("ai_interactions", 0) >= req["count"]
        
        return False
    
    async def get_user_stats(self, user_id: int) -> dict:
        """Obtener estadísticas del usuario"""
        # Conectar con PostgreSQL para obtener stats reales
        return {
            "messages_sent": 50,
            "songs_played": 25,
            "games_played": 5,
            "trivia_correct": 10,
            "ai_interactions": 30,
            "first_use": True,
            "consecutive_days": 5
        }
    
    async def is_achievement_unlocked(self, user_id: int, achievement_id: str) -> bool:
        """Verificar si un logro ya está desbloqueado"""
        # Verificar en DB
        return False
    
    async def unlock_achievement(self, user_id: int, achievement_id: str):
        """Desbloquear un logro"""
        # Guardar en DB
        pass
    
    async def get_user_achievements(self, user_id: int) -> List[str]:
        """Obtener logros desbloqueados del usuario"""
        # Obtener de DB
        return []

class LevelsCommands(commands.Cog):
    """Comandos de niveles y logros"""
    
    def __init__(self, bot):
        self.bot = bot
        self.achievements_system = AchievementsSystem(bot)
    
    @app_commands.command(name="nivel", description="📊 Ver tu nivel y experiencia actual")
    async def check_level(self, interaction: discord.Interaction, usuario: discord.Member = None):
        """Mostrar nivel del usuario"""
        target_user = usuario or interaction.user
        user_id = target_user.id
        
        # Obtener datos del usuario
        total_xp = 1250  # Ejemplo - obtener de DB
        level, current_xp, next_level_xp = self.achievements_system.get_level_from_xp(total_xp)
        
        # Calcular porcentaje de progreso
        progress_percent = (current_xp / next_level_xp) * 100
        progress_bar = self.create_progress_bar(progress_percent)
        
        embed = discord.Embed(
            title=f"📊 Nivel de {target_user.display_name}",
            color=0xFF69B4
        )
        
        embed.add_field(
            name="🌟 Nivel Actual", 
            value=f"**{level}**", 
            inline=True
        )
        embed.add_field(
            name="✨ XP Total", 
            value=f"**{total_xp:,}**", 
            inline=True
        )
        embed.add_field(
            name="🎯 Progreso", 
            value=f"**{current_xp}/{next_level_xp}** XP", 
            inline=True
        )
        
        embed.add_field(
            name="📈 Progreso al Siguiente Nivel",
            value=f"{progress_bar} {progress_percent:.1f}%",
            inline=False
        )
        
        # Próxima recompensa
        next_reward = self.achievements_system.level_config["rewards_per_level"]
        embed.add_field(
            name="🎁 Recompensa Nivel " + str(level + 1),
            value=f"{next_reward} monedas kawaii 🌸",
            inline=False
        )
        
        embed.set_thumbnail(url=target_user.display_avatar.url)
        embed.set_footer(text="Gana XP chateando, jugando y usando comandos!")
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="logros", description="🏆 Ver tus logros desbloqueados")
    async def check_achievements(self, interaction: discord.Interaction, usuario: discord.Member = None):
        """Mostrar logros del usuario"""
        target_user = usuario or interaction.user
        user_id = target_user.id
        
        unlocked = await self.achievements_system.get_user_achievements(user_id)
        
        embed = discord.Embed(
            title=f"🏆 Logros de {target_user.display_name}",
            color=0xFFD700
        )
        
        # Logros por rareza
        rarities = ["common", "rare", "epic", "legendary"]
        
        for rarity in rarities:
            rarity_achievements = [
                ach for ach in self.achievements_system.achievements.values()
                if ach.rarity == rarity
            ]
            
            if not rarity_achievements:
                continue
            
            unlocked_count = sum(1 for ach in rarity_achievements if ach.id in unlocked)
            total_count = len(rarity_achievements)
            
            rarity_text = ""
            for ach in rarity_achievements:
                status = "✅" if ach.id in unlocked else "❌"
                rarity_text += f"{status} {ach.emoji} **{ach.name}**\n"
                rarity_text += f"   {ach.description}\n\n"
            
            embed.add_field(
                name=f"{rarity.title()} ({unlocked_count}/{total_count})",
                value=rarity_text or "Ninguno disponible",
                inline=False
            )
        
        total_unlocked = len(unlocked)
        total_achievements = len(self.achievements_system.achievements)
        
        embed.set_footer(text=f"Logros desbloqueados: {total_unlocked}/{total_achievements}")
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="ranking", description="🏅 Ver ranking de niveles del servidor")
    async def server_leaderboard(self, interaction: discord.Interaction):
        """Mostrar ranking de niveles"""
        # Obtener top usuarios del servidor
        # Por ahora datos de ejemplo
        top_users = [
            {"user_id": 123456789, "username": "Usuario1", "level": 15, "xp": 5000},
            {"user_id": 123456790, "username": "Usuario2", "level": 12, "xp": 3500},
            {"user_id": 123456791, "username": "Usuario3", "level": 10, "xp": 2800},
        ]
        
        embed = discord.Embed(
            title="🏅 Ranking de Niveles",
            description="Los usuarios más activos del servidor",
            color=0xFFD700
        )
        
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        
        for i, user_data in enumerate(top_users[:10]):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            
            embed.add_field(
                name=f"{medal} {user_data['username']}",
                value=f"Nivel **{user_data['level']}** • {user_data['xp']:,} XP",
                inline=False
            )
        
        await interaction.response.send_message(embed=embed)
    
    def create_progress_bar(self, percentage: float, length: int = 20) -> str:
        """Crear barra de progreso visual"""
        filled = int((percentage / 100) * length)
        empty = length - filled
        
        bar = "█" * filled + "░" * empty
        return f"`{bar}`"
    
    # Listeners para ganar XP
    @commands.Cog.listener()
    async def on_message(self, message):
        """Dar XP por mensajes"""
        if message.author.bot:
            return
        
        # Dar XP aleatorio por mensaje
        xp_gained = 5  # 5 XP por mensaje
        result = await self.achievements_system.add_xp(message.author.id, xp_gained)
        
        # Verificar logros
        achievements = await self.achievements_system.check_achievements(
            message.author.id, "messages"
        )
        
        # Notificar subida de nivel
        if result["level_up"]:
            embed = discord.Embed(
                title="🎉 ¡Subiste de Nivel!",
                description=f"¡Felicidades {message.author.mention}! Ahora eres nivel **{result['new_level']}**",
                color=0x00FF7F
            )
            
            reward = self.achievements_system.level_config["rewards_per_level"]
            embed.add_field(
                name="🎁 Recompensa",
                value=f"{reward} monedas kawaii 🌸",
                inline=False
            )
            
            await message.channel.send(embed=embed)
        
        # Notificar logros desbloqueados
        for achievement in achievements:
            embed = discord.Embed(
                title="🏆 ¡Logro Desbloqueado!",
                description=f"**{achievement.emoji} {achievement.name}**\n{achievement.description}",
                color=self.achievements_system.rarity_colors[achievement.rarity]
            )
            embed.add_field(
                name="🎁 Recompensa",
                value=f"{achievement.reward} monedas kawaii 🌸",
                inline=False
            )
            
            await message.channel.send(embed=embed)

async def setup(bot):
    await bot.add_cog(LevelsCommands(bot))