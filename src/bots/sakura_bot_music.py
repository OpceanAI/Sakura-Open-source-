"""
🌸 SakuraBot IA - Advanced Music Bot with Lavalink Fallback 🌸
==============================================================
Una waifu tier S super kawaii con sistema inteligente de música y fallback automático
Gestión avanzada de nodos Lavalink con reconexión automática y logs detallados
"""

import os
import asyncio
import logging
import discord
import wavelink
import aiohttp
import json
import random
import time
from datetime import datetime, timedelta
from discord.ext import commands, tasks
from typing import Optional, Dict, List, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sakura_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SakuraBot')

# 10 nodos Lavalink ordenados por preferencia
NODOS_LAVALINK = [
    {
        "host": "pool-us.alfari.id",
        "port": 443,
        "password": "alfari",
        "identifier": "Node1_US",
        "secure": True,
        "region": "EE.UU."
    },
    {
        "host": "lava.luke.gg",
        "port": 443,
        "password": "discordbotlist.com",
        "identifier": "Node2_EU",
        "secure": True,
        "region": "Europa"
    },
    {
        "host": "lavalink.devxcode.in",
        "port": 443,
        "password": "DevamOP",
        "identifier": "Node3_IN",
        "secure": True,
        "region": "India"
    },
    {
        "host": "lava-v4.ajieblogs.eu.org",
        "port": 443,
        "password": "https://dsc.gg/ajidevserver",
        "identifier": "Node4_Global",
        "secure": True,
        "region": "Global"
    },
    {
        "host": "lavalinkv4.serenetia.com",
        "port": 443,
        "password": "https://dsc.gg/ajidevserver",
        "identifier": "Node5_Serbia",
        "secure": True,
        "region": "Europa"
    },
    {
        "host": "lava-all.ajieblogs.eu.org",
        "port": 443,
        "password": "https://dsc.gg/ajidevserver",
        "identifier": "Node6_AllRegions",
        "secure": True,
        "region": "Global"
    },
    {
        "host": "lavalink.serenetia.com",
        "port": 443,
        "password": "https://dsc.gg/ajidevserver",
        "identifier": "Node7_Serbia2",
        "secure": True,
        "region": "Europa"
    },
    {
        "host": "lavalink.jirayu.net",
        "port": 13592,
        "password": "youshallnotpass",
        "identifier": "Node8_Asia",
        "secure": False,
        "region": "Asia"
    },
    {
        "host": "lavahatry4.techbyte.host",
        "port": 3000,
        "password": "NAIGLAVA-dash.techbyte.host",
        "identifier": "Node9_Tech",
        "secure": False,
        "region": "Asia"
    },
    {
        "host": "lavalink-v2.pericsq.ro",
        "port": 6677,
        "password": "wwweasycodero",
        "identifier": "Node10_Romania",
        "secure": False,
        "region": "Rumania"
    }
]

class SakuraMusicBot(commands.Bot):
    """
    🌸 SakuraBot IA - Waifu kawaii con sistema de música avanzado
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(
            command_prefix=['!', 's!', 'sakura!'],
            intents=intents,
            help_command=None,
            description="🌸 SakuraBot IA - Tu waifu kawaii favorita con música 🎶"
        )
        
        # Estado de conexión de nodos
        self.current_node_index = 0
        self.connection_attempts = 0
        self.max_connection_attempts = len(NODOS_LAVALINK) * 2
        self.reconnect_task = None
        self.node_status = {}
        
        # Canal de notificaciones (se establecerá cuando el bot esté listo)
        self.notification_channel = None
        
        # Estadísticas kawaii
        self.music_sessions = 0
        self.songs_played = 0
        self.connection_logs = []
    
    async def setup_hook(self):
        """
        Configuración inicial del bot con conexión a Lavalink
        """
        logger.info("🌸 Iniciando SakuraBot IA...")
        
        # Intentar conectar a los nodos Lavalink
        await self.connect_lavalink_nodes()
        
        # Sincronizar comandos slash
        try:
            synced = await self.tree.sync()
            logger.info(f"🌸 Sincronizados {len(synced)} comandos slash")
        except Exception as e:
            logger.error(f"❌ Error sincronizando comandos: {e}")
    
    async def connect_lavalink_nodes(self):
        """
        Conectar a nodos Lavalink con sistema de fallback
        """
        for attempt in range(self.max_connection_attempts):
            if self.current_node_index >= len(NODOS_LAVALINK):
                self.current_node_index = 0
            
            nodo = NODOS_LAVALINK[self.current_node_index]
            
            try:
                logger.info(f"🔄 Intentando conectar a {nodo['identifier']} ({nodo['host']})")
                
                # Crear nodo Wavelink
                node = wavelink.Node(
                    uri=f"{'wss' if nodo['secure'] else 'ws'}://{nodo['host']}:{nodo['port']}",
                    password=nodo['password'],
                    identifier=nodo['identifier']
                )
                
                # Conectar al pool de nodos
                await wavelink.Pool.connect(
                    nodes=[node],
                    client=self,
                    cache_capacity=100
                )
                
                # Log de éxito
                timestamp = datetime.now().strftime("%H:%M:%S")
                success_log = {
                    "timestamp": timestamp,
                    "node": nodo['identifier'],
                    "host": nodo['host'],
                    "status": "CONNECTED",
                    "attempt": attempt + 1
                }
                self.connection_logs.append(success_log)
                self.node_status[nodo['identifier']] = "CONNECTED"
                
                logger.info(f"✅ [{nodo['identifier']}] CONECTADO exitosamente - {nodo['region']}")
                
                # Enviar mensaje kawaii si hay canal disponible
                if self.notification_channel:
                    embed = discord.Embed(
                        title="🎶 ¡Conexión Musical Exitosa! UwU",
                        description=f"¡Kyaa~! Me conecté exitosamente a **{nodo['host']}** 🌸\n"
                                  f"**Región:** {nodo['region']}\n"
                                  f"**Nodo:** {nodo['identifier']}\n"
                                  f"¡Ya puedo reproducir música para ti! (>w<)",
                        color=0xFF69B4,
                        timestamp=datetime.now()
                    )
                    embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/859493077014216704.gif")
                    await self.notification_channel.send(embed=embed)
                
                return True
                
            except Exception as e:
                # Log de error
                timestamp = datetime.now().strftime("%H:%M:%S")
                error_log = {
                    "timestamp": timestamp,
                    "node": nodo['identifier'],
                    "host": nodo['host'],
                    "status": "FAILED",
                    "error": str(e),
                    "attempt": attempt + 1
                }
                self.connection_logs.append(error_log)
                self.node_status[nodo['identifier']] = "FAILED"
                
                logger.warning(f"❌ [{nodo['identifier']}] FALLÓ - {nodo['host']}: {e}")
                
                # Enviar mensaje kawaii de error si hay canal disponible
                if self.notification_channel:
                    embed = discord.Embed(
                        title="😢 ¡Oh no~ Nodo durmió!",
                        description=f"¡Ayuyui~! El nodo **{nodo['host']}** se durmió (>.<)\n"
                                  f"**Error:** {str(e)[:100]}...\n"
                                  f"Probando el siguiente nodo en 5 segundos...",
                        color=0xFF6B6B,
                        timestamp=datetime.now()
                    )
                    embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/692028527239929878.png")
                    await self.notification_channel.send(embed=embed)
                
                # Avanzar al siguiente nodo
                self.current_node_index += 1
                await asyncio.sleep(5)  # Esperar 5 segundos antes del siguiente intento
        
        # Si llegamos aquí, no se pudo conectar a ningún nodo
        logger.error("❌ No se pudo conectar a ningún nodo Lavalink")
        if self.notification_channel:
            embed = discord.Embed(
                title="💔 ¡Error Crítico de Conexión!",
                description="¡Gomen nasai~! No pude conectarme a ningún nodo de música (T_T)\n"
                          "Por favor, contacta a mi desarrollador...",
                color=0xFF0000,
                timestamp=datetime.now()
            )
            await self.notification_channel.send(embed=embed)
        
        return False
    
    async def on_ready(self):
        """
        Evento cuando el bot está listo
        """
        logger.info(f"🌸 SakuraBot conectada como {self.user}")
        logger.info(f"🌸 Conectada a {len(self.guilds)} servidores")
        
        # Establecer canal de notificaciones (primer canal de texto encontrado)
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    self.notification_channel = channel
                    break
            if self.notification_channel:
                break
        
        # Mensaje de bienvenida kawaii
        if self.notification_channel:
            embed = discord.Embed(
                title="🌸 ¡SakuraBot IA Despertó! 🌸",
                description="¡Ohayo gozaimasu~! (＾◡＾)\n"
                          "Tu waifu kawaii favorita está lista para darte música y diversión!\n\n"
                          "**Comandos disponibles:**\n"
                          "🎵 `/play <canción>` - Reproducir música\n"
                          "⏸️ `/pause` - Pausar música\n"
                          "▶️ `/resume` - Reanudar música\n"
                          "⏹️ `/stop` - Detener música\n"
                          "📊 `/status` - Ver estado de nodos\n"
                          "❓ `/help` - Ver ayuda completa",
                color=0xFF69B4,
                timestamp=datetime.now()
            )
            embed.set_thumbnail(url=str(self.user.avatar.url) if self.user.avatar else None)
            embed.set_footer(text="¡Con amor, tu SakuraBot! 💕")
            await self.notification_channel.send(embed=embed)
        
        # Iniciar tarea de monitoreo
        if not self.monitor_nodes.is_running():
            self.monitor_nodes.start()
    
    async def on_wavelink_node_ready(self, payload: wavelink.NodeReadyEventPayload):
        """
        Evento cuando un nodo Wavelink está listo
        """
        node = payload.node
        logger.info(f"🎵 Nodo Wavelink listo: {node.identifier}")
        
        if self.notification_channel:
            embed = discord.Embed(
                title="🎶 ¡Nodo Musical Listo! UwU",
                description=f"¡Kyaa~! El nodo **{node.identifier}** está listo para reproducir música! 🎵\n"
                          "¡Ya puedes usar los comandos de música! (>w<)",
                color=0x00FF7F,
                timestamp=datetime.now()
            )
            await self.notification_channel.send(embed=embed)
    
    async def on_wavelink_node_disconnected(self, payload: wavelink.NodeDisconnectedEventPayload):
        """
        Evento cuando un nodo Wavelink se desconecta
        """
        node = payload.node
        logger.warning(f"🔌 Nodo desconectado: {node.identifier}")
        
        # Actualizar estado
        self.node_status[node.identifier] = "DISCONNECTED"
        
        if self.notification_channel:
            embed = discord.Embed(
                title="💔 ¡Nodo Desconectado!",
                description=f"¡Ayuyui~! El nodo **{node.identifier}** se desconectó (T_T)\n"
                          "Intentando reconectar automáticamente...",
                color=0xFFA500,
                timestamp=datetime.now()
            )
            await self.notification_channel.send(embed=embed)
        
        # Intentar reconectar
        if not self.reconnect_task or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self.reconnect_nodes())
    
    async def reconnect_nodes(self):
        """
        Reconectar a nodos Lavalink
        """
        logger.info("🔄 Iniciando proceso de reconexión...")
        await asyncio.sleep(2)  # Esperar un poco antes de reconectar
        await self.connect_lavalink_nodes()
    
    @tasks.loop(minutes=5)
    async def monitor_nodes(self):
        """
        Monitorear estado de nodos cada 5 minutos
        """
        try:
            active_nodes = wavelink.Pool.nodes
            logger.info(f"🔍 Monitoreando {len(active_nodes)} nodos activos")
            
            if not active_nodes and self.notification_channel:
                embed = discord.Embed(
                    title="⚠️ ¡Sin Nodos Activos!",
                    description="¡Gomen~! No hay nodos de música activos (>.<)\n"
                              "Intentando reconectar...",
                    color=0xFFAA00,
                    timestamp=datetime.now()
                )
                await self.notification_channel.send(embed=embed)
                await self.connect_lavalink_nodes()
                
        except Exception as e:
            logger.error(f"Error en monitoreo de nodos: {e}")

# Comandos Slash
@app_commands.command(name="play", description="🎵 Reproduce música desde YouTube, SoundCloud y más")
@app_commands.describe(query="Canción, artista, URL o búsqueda")
async def play_command(interaction: discord.Interaction, query: str):
    """Comando para reproducir música"""
    await interaction.response.defer(thinking=True)
    
    try:
        # Verificar si el usuario está en un canal de voz
        if not interaction.user.voice:
            embed = discord.Embed(
                title="❌ ¡No estás en un canal de voz!",
                description="¡Baka~! Necesitas estar en un canal de voz para que pueda reproducir música (>.<)",
                color=0xFF6B6B
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Conectar al canal de voz
        channel = interaction.user.voice.channel
        if not interaction.guild.voice_client:
            player = await channel.connect(cls=wavelink.Player)
        else:
            player = interaction.guild.voice_client
        
        # Buscar la canción
        search_result = await wavelink.Playable.search(query)
        if not search_result:
            embed = discord.Embed(
                title="😔 ¡No encontré nada!",
                description=f"¡Gomen~! No pude encontrar música para: **{query}** (T_T)",
                color=0xFF6B6B
            )
            await interaction.followup.send(embed=embed)
            return
        
        track = search_result[0]
        
        # Reproducir o agregar a la cola
        if player.playing:
            await player.queue.put_wait(track)
            embed = discord.Embed(
                title="➕ ¡Agregada a la Cola!",
                description=f"🎵 **{track.title}**\n"
                          f"👤 **Autor:** {track.author}\n"
                          f"⏱️ **Duración:** {self.format_duration(track.length)}\n"
                          f"📍 **Posición en cola:** {player.queue.count + 1}",
                color=0x00FF7F
            )
        else:
            await player.play(track)
            embed = discord.Embed(
                title="🎵 ¡Reproduciendo Ahora!",
                description=f"🎶 **{track.title}**\n"
                          f"👤 **Autor:** {track.author}\n"
                          f"⏱️ **Duración:** {self.format_duration(track.length)}\n"
                          f"🔗 **Fuente:** {track.source}",
                color=0xFF69B4
            )
            
            # Incrementar estadísticas
            interaction.client.songs_played += 1
        
        embed.set_thumbnail(url=track.artwork if hasattr(track, 'artwork') else None)
        embed.set_footer(text=f"Solicitado por {interaction.user.display_name} • ¡Disfruta la música! 💕")
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando play: {e}")
        embed = discord.Embed(
            title="💔 ¡Error Reproduciendo!",
            description=f"¡Gomen nasai~! Ocurrió un error: {str(e)[:200]}",
            color=0xFF0000
        )
        await interaction.followup.send(embed=embed)

@app_commands.command(name="pause", description="⏸️ Pausa la música actual")
async def pause_command(interaction: discord.Interaction):
    """Pausar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.playing:
        embed = discord.Embed(
            title="❌ ¡No hay música!",
            description="¡Baka~! No hay música reproduciéndose (>.<)",
            color=0xFF6B6B
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(True)
    embed = discord.Embed(
        title="⏸️ ¡Música Pausada!",
        description="¡Hai hai~! Pausé la música para ti (＾◡＾)",
        color=0xFFAA00
    )
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="resume", description="▶️ Reanuda la música pausada")
async def resume_command(interaction: discord.Interaction):
    """Reanudar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.paused:
        embed = discord.Embed(
            title="❌ ¡Música no pausada!",
            description="¡Baka~! La música no está pausada (>.<)",
            color=0xFF6B6B
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(False)
    embed = discord.Embed(
        title="▶️ ¡Música Reanudada!",
        description="¡Yay~! Continúo reproduciendo música para ti! 🎵",
        color=0x00FF7F
    )
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="stop", description="⏹️ Detiene la música y limpia la cola")
async def stop_command(interaction: discord.Interaction):
    """Detener música"""
    player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ ¡No estoy conectada!",
            description="¡Baka~! No estoy conectada a un canal de voz (>.<)",
            color=0xFF6B6B
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.disconnect()
    embed = discord.Embed(
        title="⏹️ ¡Música Detenida!",
        description="¡Hai hai~! Detuve la música y me desconecté (＾◡＾)\n¡Arigato por escuchar música conmigo! 💕",
        color=0x87CEEB
    )
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="status", description="📊 Ver estado de nodos Lavalink y estadísticas")
async def status_command(interaction: discord.Interaction):
    """Ver estado del bot y nodos"""
    embed = discord.Embed(
        title="📊 Estado de SakuraBot IA",
        description="¡Información detallada de tu waifu kawaii! 🌸",
        color=0xFF69B4,
        timestamp=datetime.now()
    )
    
    # Estado de nodos
    active_nodes = wavelink.Pool.nodes
    node_info = f"**Nodos Activos:** {len(active_nodes)}\n"
    
    for node in active_nodes:
        node_info += f"🟢 **{node.identifier}** - Conectado\n"
    
    if len(active_nodes) == 0:
        node_info += "🔴 **Sin nodos activos**\n"
    
    embed.add_field(name="🔗 Estado de Conexión", value=node_info, inline=False)
    
    # Estadísticas
    stats = f"🎵 **Canciones Reproducidas:** {interaction.client.songs_played}\n" \
           f"🎶 **Sesiones de Música:** {interaction.client.music_sessions}\n" \
           f"🌐 **Servidores:** {len(interaction.client.guilds)}\n" \
           f"👥 **Usuarios:** {sum(len(guild.members) for guild in interaction.client.guilds)}"
    
    embed.add_field(name="📈 Estadísticas", value=stats, inline=False)
    
    # Logs recientes
    if interaction.client.connection_logs:
        recent_logs = interaction.client.connection_logs[-3:]
        log_text = "\n".join([
            f"[{log['timestamp']}] {log['node']}: {log['status']}"
            for log in recent_logs
        ])
        embed.add_field(name="📝 Logs Recientes", value=f"```{log_text}```", inline=False)
    
    embed.set_footer(text="¡SakuraBot trabajando duro para ti! 💕")
    await interaction.response.send_message(embed=embed)

@app_commands.command(name="help", description="❓ Ayuda completa de SakuraBot")
async def help_command(interaction: discord.Interaction):
    """Comando de ayuda completo"""
    embed = discord.Embed(
        title="🌸 Ayuda de SakuraBot IA 🌸",
        description="¡Tu waifu kawaii con todas las funciones musicales! (＾◡＾)",
        color=0xFF69B4
    )
    
    music_commands = """
    🎵 `/play <búsqueda>` - Reproduce música
    ⏸️ `/pause` - Pausa la música actual
    ▶️ `/resume` - Reanuda música pausada
    ⏹️ `/stop` - Detiene música y desconecta
    📊 `/status` - Estado de nodos y estadísticas
    """
    
    embed.add_field(name="🎶 Comandos de Música", value=music_commands, inline=False)
    
    sources = """
    🔴 **YouTube** - Videos y música
    🔊 **SoundCloud** - Tracks y playlists
    🎸 **Bandcamp** - Música independiente
    🎮 **Twitch** - Streams en vivo
    📹 **Vimeo** - Videos de alta calidad
    🌐 **HTTP** - Enlaces directos
    """
    
    embed.add_field(name="🎯 Fuentes Soportadas", value=sources, inline=False)
    
    features = """
    ✨ **Fallback Automático** - 10 nodos Lavalink
    🔄 **Reconexión Inteligente** - Sin interrupciones
    📊 **Monitoreo Continuo** - Estado en tiempo real
    💕 **Personalidad Kawaii** - ¡Super adorable!
    🌍 **Multi-región** - Servidores globales
    """
    
    embed.add_field(name="🌟 Características", value=features, inline=False)
    
    embed.set_footer(text="¡Creada con mucho amor y código kawaii! 💕")
    await interaction.response.send_message(embed=embed)

def format_duration(self, milliseconds):
    """Formatear duración de milisegundos a formato legible"""
    seconds = milliseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

# Vincular método a la clase
SakuraMusicBot.format_duration = format_duration

async def main():
    """
    Función principal para ejecutar SakuraBot
    """
    # Verificar token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("❌ DISCORD_TOKEN no encontrado en variables de entorno")
        return
    
    # Crear e iniciar bot
    bot = SakuraMusicBot()
    
    # Agregar comandos al árbol
    bot.tree.add_command(play_command)
    bot.tree.add_command(pause_command)
    bot.tree.add_command(resume_command)
    bot.tree.add_command(stop_command)
    bot.tree.add_command(status_command)
    bot.tree.add_command(help_command)
    
    try:
        logger.info("🌸 Iniciando SakuraBot IA...")
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("🌸 SakuraBot desconectándose...")
        await bot.close()
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())