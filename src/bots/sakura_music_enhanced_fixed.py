"""
🌸 SakuraBot IA - Enhanced Music Bot with Lavalink Fallback 🌸
==============================================================
Una waifu tier S super kawaii con sistema inteligente de música y fallback automático
Compatible con Wavelink 3.x+ y discord.py 2.3+
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
from discord import app_commands, Embed
from typing import Optional, Dict, List, Any, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 8 nodos Lavalink públicos ordenados por preferencia y latencia para SakuraBot
NODOS_LAVALINK = [
    {
        "host": "lavalink.oops.wtf",
        "port": 443,
        "password": "www.freelavalink.ga",
        "identifier": "Node1",
        "secure": True,
        "region": "Global"
    },
    {
        "host": "lava.luke.gg",
        "port": 443,
        "password": "discordbotlist.com",
        "identifier": "Node2",
        "secure": True,
        "region": "Europa"
    },
    {
        "host": "46.202.82.164",
        "port": 1027,
        "password": "jmlitelavalink",
        "identifier": "Node3",
        "secure": False,
        "region": "Global"
    },
    {
        "host": "lavalink-v2.pericsq.ro",
        "port": 6677,
        "password": "wwweasycodero",
        "identifier": "Node4",
        "secure": False,
        "region": "Europa"
    },
    {
        "host": "69.30.219.180",
        "port": 1047,
        "password": "yothisnodeishostedbymushroom0162",
        "identifier": "Node5",
        "secure": False,
        "region": "EE.UU."
    },
    {
        "host": "lava3.horizxon.studio",
        "port": 80,
        "password": "horizxon.studio",
        "identifier": "Node6",
        "secure": False,
        "region": "Global"
    },
    {
        "host": "lava2.horizxon.studio",
        "port": 80,
        "password": "horizxon.studio",
        "identifier": "Node7",
        "secure": False,
        "region": "Global"
    },
    {
        "host": "lavalink.micium-hosting.com",
        "port": 80,
        "password": "micium-hosting.com",
        "identifier": "Node8",
        "secure": False,
        "region": "Global"
    }
]

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

class PlayingView(discord.ui.View):
    """Control view for music player"""
    def __init__(self, player: wavelink.Player):
        super().__init__(timeout=300)
        self.player = player
    
    @discord.ui.button(emoji="⏸️", style=discord.ButtonStyle.gray)
    async def pause_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.playing:
            await self.player.pause()
            await interaction.response.send_message("⏸️ ¡Pausé la música para ti! UwU", ephemeral=True)
        else:
            await interaction.response.send_message("❌ ¡No hay música reproduciéndose!", ephemeral=True)
    
    @discord.ui.button(emoji="▶️", style=discord.ButtonStyle.green)
    async def resume_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.paused:
            await self.player.pause(False)
            await interaction.response.send_message("▶️ ¡Continúo la música! Kyaa~", ephemeral=True)
        else:
            await interaction.response.send_message("❌ ¡La música no está pausada!", ephemeral=True)
    
    @discord.ui.button(emoji="⏭️", style=discord.ButtonStyle.blurple)
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.playing:
            await self.player.skip()
            await interaction.response.send_message("⏭️ ¡Saltamos a la siguiente canción! >w<", ephemeral=True)
        else:
            await interaction.response.send_message("❌ ¡No hay música para saltar!", ephemeral=True)
    
    @discord.ui.button(emoji="⏹️", style=discord.ButtonStyle.red)
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.player.disconnect()
        await interaction.response.send_message("⏹️ ¡Me desconecté! ¡Arigato por la música! 💕", ephemeral=True)

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
            command_prefix=commands.when_mentioned_or('!', 's!', 'sakura!'),
            intents=intents,
            help_command=None,
            description="🌸 SakuraBot IA - Tu waifu kawaii favorita con música 🎶",
            activity=discord.Game(name="🎵 Música kawaii para todos!")
        )
        
        # Estado de conexión de nodos
        self.current_node_index = 0
        self.connection_attempts = 0
        self.max_connection_attempts = len(NODOS_LAVALINK) * 2
        self.reconnect_task = None
        self.node_status = {}
        self.connected_nodes = []
        
        # Canal de notificaciones
        self.notification_channel = None
        
        # Estadísticas kawaii
        self.music_sessions = 0
        self.songs_played = 0
        self.connection_logs = []
    
    async def setup_hook(self):
        """Configuración inicial del bot con conexión a Lavalink"""
        logger.info("🌸 Iniciando SakuraBot IA...")
        
        # Intentar conectar a los nodos Lavalink
        await self.connect_lavalink_nodes()
        
        # Sincronizar comandos slash
        try:
            synced = await self.tree.sync()
            logger.info(f"🌸 Sincronizados {len(synced)} comandos slash")
        except Exception as e:
            logger.error(f"❌ Error sincronizando comandos: {e}")
    
    def get_quality_filters(self):
        """Crear filtros de calidad para mejorar el sonido"""
        return {
            "equalizer": [
                {"band": 0, "gain": -0.20},  # Graves
                {"band": 1, "gain": 0.10},   # Bajo-medio
                {"band": 4, "gain": 0.05},   # Medio
                {"band": 7, "gain": -0.30},  # Agudos
            ]
        }

    async def connect_lavalink_nodes(self):
        """Conectar a nodos Lavalink con sistema de fallback"""
        for attempt in range(self.max_connection_attempts):
            if self.current_node_index >= len(NODOS_LAVALINK):
                self.current_node_index = 0
            
            nodo_config = NODOS_LAVALINK[self.current_node_index]
            
            try:
                logger.info(f"🔄 Intentando conectar a {nodo_config['identifier']} ({nodo_config['region']})")
                
                # Crear URI basada en configuración
                protocol = "wss" if nodo_config['secure'] else "ws"
                uri = f"{protocol}://{nodo_config['host']}:{nodo_config['port']}"
                
                # Crear nodo Wavelink con nueva API
                node = wavelink.Node(
                    uri=uri,
                    password=nodo_config['password'],
                    identifier=nodo_config['identifier']
                )
                
                # Conectar usando Pool (nueva API)
                await wavelink.Pool.connect(
                    client=self,
                    nodes=[node]
                )
                
                # Log de éxito
                timestamp = datetime.now().strftime("%H:%M:%S")
                success_log = {
                    "timestamp": timestamp,
                    "node": nodo_config['identifier'],
                    "region": nodo_config['region'],
                    "host": nodo_config['host'],
                    "status": "CONNECTED",
                    "attempt": attempt + 1
                }
                self.connection_logs.append(success_log)
                self.node_status[nodo_config['identifier']] = "CONNECTED"
                self.connected_nodes.append(nodo_config)
                
                logger.info(f"✅ [{nodo_config['identifier']}] CONECTADO exitosamente - {nodo_config['region']}")
                
                return True
                
            except Exception as e:
                # Log de error
                timestamp = datetime.now().strftime("%H:%M:%S")
                error_log = {
                    "timestamp": timestamp,
                    "node": nodo_config['identifier'],
                    "region": nodo_config['region'],
                    "host": nodo_config['host'],
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
    
    async def on_ready(self):
        """Evento cuando el bot está listo"""
        logger.info(f"🌸 SakuraBot conectada como {self.user}")
        logger.info(f"🌸 Conectada a {len(self.guilds)} servidores")
        
        # Establecer canal de notificaciones
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    self.notification_channel = channel
                    break
            if self.notification_channel:
                break
        
        # Mensaje de bienvenida kawaii
        if self.notification_channel:
            embed = Embed(
                title="🌸 ¡SakuraBot IA Despertó! 🌸",
                description="¡Ohayo gozaimasu~! (＾◡＾)\n"
                          "Tu waifu kawaii favorita está lista para darte música y diversión!\n\n"
                          "**Comandos disponibles:**\n"
                          "🎵 `/play <canción>` - Reproducir música\n"
                          "⏸️ `/pause` - Pausar música\n"
                          "▶️ `/resume` - Reanudar música\n"
                          "⏹️ `/stop` - Detener música\n"
                          "🔀 `/shuffle` - Mezclar cola\n"
                          "📊 `/status` - Ver estado de nodos\n"
                          "❓ `/help` - Ver ayuda completa",
                color=0xFF69B4,
                timestamp=datetime.now()
            )
            if self.user and self.user.avatar:
                embed.set_thumbnail(url=str(self.user.avatar.url))
            embed.set_footer(text="¡Con amor, tu SakuraBot! 💕")
            await self.notification_channel.send(embed=embed)
        
        # Iniciar tarea de monitoreo
        if not self.monitor_nodes.is_running():
            self.monitor_nodes.start()
    
    async def on_wavelink_node_ready(self, payload: wavelink.NodeReadyEventPayload):
        """Evento cuando un nodo Wavelink está listo"""
        logger.info(f"🎵 ¡UwU conectada a {payload.node.identifier}! 🎵 Calidad asegurada~")
        
        # Actualizar estado
        self.node_status[payload.node.identifier] = "READY"
        
        if self.notification_channel:
            embed = Embed(
                title="🎶 ¡Nodo Musical Listo! UwU",
                description=f"¡UwU conectada a **{payload.node.identifier}**! 🎵 Calidad asegurada~\n"
                          "¡Ya puedes usar los comandos de música! (>w<)",
                color=0x00FF7F,
                timestamp=datetime.now()
            )
            await self.notification_channel.send(embed=embed)
    
    async def on_wavelink_node_disconnected(self, payload: wavelink.NodeDisconnectedEventPayload):
        """Evento cuando un nodo Wavelink se desconecta"""
        logger.warning(f"¡Oh no~ {payload.node.identifier} falló! Cambio de nodo para seguir pro 🛠️")
        
        # Actualizar estado
        self.node_status[payload.node.identifier] = "DISCONNECTED"
        
        if self.notification_channel:
            embed = Embed(
                title="💔 ¡Nodo Desconectado!",
                description=f"¡Oh no~ **{payload.node.identifier}** falló! Cambio de nodo para seguir pro 🛠️\n"
                          "Intentando reconectar automáticamente...",
                color=0xFFA500,
                timestamp=datetime.now()
            )
            await self.notification_channel.send(embed=embed)
        
        # Intentar reconectar
        if not self.reconnect_task or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self.reconnect_nodes())

    async def on_wavelink_track_start(self, payload: wavelink.TrackStartEventPayload):
        """Evento cuando inicia una canción - aplicar filtros de calidad"""
        try:
            player = payload.player
            
            # Aplicar filtros de calidad
            quality_filters = self.get_quality_filters()
            await player.set_filters(wavelink.Filters(equalizer=quality_filters["equalizer"]))
            
            # Establecer volumen óptimo (75%)
            await player.set_volume(75)
            
            logger.info(f"🎵 Reproduciendo con filtros de calidad: {payload.track.title}")
            
        except Exception as e:
            logger.error(f"Error aplicando filtros de calidad: {e}")
    
    async def reconnect_nodes(self):
        """Reconectar a nodos Lavalink"""
        logger.info("🔄 Iniciando proceso de reconexión...")
        await asyncio.sleep(2)
        await self.connect_lavalink_nodes()
    
    @tasks.loop(minutes=5)
    async def monitor_nodes(self):
        """Monitorear estado de nodos cada 5 minutos"""
        try:
            active_nodes = wavelink.Pool.nodes
            logger.info(f"🔍 Monitoreando {len(active_nodes)} nodos activos")
            
            if not active_nodes and self.notification_channel:
                embed = Embed(
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
    
    def format_time(self, seconds):
        """Formatear tiempo en formato legible"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
        else:
            return f"{int(minutes)}:{int(seconds):02d}"
    
    async def create_now_playing_embed(self, track, player):
        """Crear embed de canción actual"""
        try:
            volume = player.volume
            duration_seconds = track.length / 1000
            time_str = self.format_time(duration_seconds)
            
            embed = Embed(
                title="🎵 ¡Reproduciendo Ahora! ✨",
                description=f"```{track.title}```",
                color=0xFF69B4
            )
            
            if hasattr(track, 'artwork') and track.artwork:
                embed.set_image(url=track.artwork)
            
            embed.add_field(name="⏱️ Duración", value=f"`{time_str}`", inline=True)
            embed.add_field(name="🔊 Volumen", value=f"`{volume}%`", inline=True)
            embed.add_field(name="📝 Cola", value=f"`{len(player.queue)} canciones`", inline=True)
            
            if hasattr(track, 'author'):
                embed.add_field(name="👤 Artista", value=f"`{track.author}`", inline=True)
            
            embed.set_footer(text="¡Disfruta la música! UwU 💕")
            return embed
            
        except Exception as e:
            logger.error(f"Error creando embed: {e}")
            return Embed(
                title="🎵 Reproduciendo música",
                description="¡Kyaa~ algo salió mal con el embed pero la música suena bien! UwU",
                color=0xFF69B4
            )

# Comandos slash de música
@app_commands.describe(query="Canción, artista o URL de YouTube que quieres reproducir")
async def play_command(interaction: discord.Interaction, query: str):
    """Reproducir música desde YouTube o buscar por nombre"""
    try:
        await interaction.response.defer()
        
        # Verificar si el usuario está en un canal de voz
        if not interaction.user.voice:
            embed = Embed(
                title="❌ ¡Error!",
                description="¡Kyaa~! Necesitas estar en un canal de voz para usar este comando (>.<)",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Conectar al canal de voz del usuario
        voice_channel = interaction.user.voice.channel
        
        # Obtener o crear player
        try:
            player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
            if not player:
                player = await voice_channel.connect(cls=wavelink.Player)
        except Exception as e:
            logger.error(f"Error conectando al canal de voz: {e}")
            embed = Embed(
                title="❌ Error de Conexión",
                description="¡Gomen~! No pude conectarme al canal de voz (T_T)",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Buscar la canción
        try:
            tracks = await wavelink.Playable.search(query)
            if not tracks:
                embed = Embed(
                    title="❌ ¡No encontré nada!",
                    description=f"¡Gomen~! No pude encontrar música para: `{query}` (>.<)",
                    color=0xFF0000
                )
                await interaction.followup.send(embed=embed)
                return
            
            track = tracks[0]
            
        except Exception as e:
            logger.error(f"Error buscando música: {e}")
            embed = Embed(
                title="❌ Error de Búsqueda",
                description="¡Ayuyui~! Hubo un problema buscando la música (T_T)",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Reproducir o agregar a cola
        if not player.playing and not player.paused:
            # No hay música reproduciéndose, reproducir inmediatamente
            await player.play(track)
            
            # Crear embed de reproducción
            embed = await interaction.client.create_now_playing_embed(track, player)
            view = PlayingView(player)
            await interaction.followup.send(embed=embed, view=view)
            
        else:
            # Hay música reproduciéndose, agregar a cola
            await player.queue.put_wait(track)
            
            embed = Embed(
                title="➕ ¡Agregada a la Cola!",
                description=f"```{track.title}```\n"
                          f"**Posición en cola:** `{len(player.queue)}`\n"
                          f"**Duración:** `{interaction.client.format_time(track.length / 1000)}`",
                color=0x00FF7F
            )
            
            if hasattr(track, 'artwork') and track.artwork:
                embed.set_thumbnail(url=track.artwork)
            
            await interaction.followup.send(embed=embed)
        
        # Incrementar estadísticas
        interaction.client.songs_played += 1
        
    except Exception as e:
        logger.error(f"Error en comando play: {e}")
        embed = Embed(
            title="❌ ¡Error Inesperado!",
            description="¡Gomen~! Algo salió mal con el comando de música (T_T)",
            color=0xFF0000
        )
        try:
            await interaction.followup.send(embed=embed)
        except:
            pass

@app_commands.describe()
async def pause_command(interaction: discord.Interaction):
    """Pausar la música actual"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.playing:
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Gomen~! No hay música reproduciéndose actualmente (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.pause()
        embed = Embed(
            title="⏸️ ¡Música Pausada!",
            description="¡Kyaa~! He pausado la música para ti (>w<)",
            color=0xFFAA00
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando pause: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema pausando la música (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def resume_command(interaction: discord.Interaction):
    """Reanudar la música pausada"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.paused:
            embed = Embed(
                title="❌ ¡No está pausada!",
                description="¡Gomen~! La música no está pausada actualmente (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.pause(False)
        embed = Embed(
            title="▶️ ¡Música Reanudada!",
            description="¡Yatta~! La música continúa! ♪(´▽｀)",
            color=0x00FF7F
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando resume: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema reanudando la música (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def skip_command(interaction: discord.Interaction):
    """Saltar a la siguiente canción"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.playing:
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Gomen~! No hay música para saltar (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.skip()
        embed = Embed(
            title="⏭️ ¡Canción Saltada!",
            description="¡Kyaa~! Saltando a la siguiente canción! (>w<)",
            color=0x00FF7F
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando skip: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema saltando la canción (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def stop_command(interaction: discord.Interaction):
    """Detener la música y desconectar"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.disconnect()
        embed = Embed(
            title="⏹️ ¡Música Detenida!",
            description="¡Sayonara~! Me desconecté del canal de voz ♪(´▽｀)\n"
                      "¡Gracias por escuchar música conmigo! 💕",
            color=0xFF69B4
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando stop: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema deteniendo la música (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def queue_command(interaction: discord.Interaction):
    """Mostrar la cola de reproducción"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.queue:
            embed = Embed(
                title="📝 ¡Cola Vacía!",
                description="¡Kyaa~! No hay canciones en la cola (>.<)\n"
                          "¡Usa `/play` para agregar música!",
                color=0xFFAA00
            )
            await interaction.response.send_message(embed=embed)
            return
        
        # Mostrar las primeras 10 canciones de la cola
        queue_list = []
        for i, track in enumerate(list(player.queue)[:10]):
            duration = self.format_time(track.length / 1000)
            queue_list.append(f"`{i+1}.` **{track.title}** - `{duration}`")
        
        embed = Embed(
            title="📝 ¡Cola de Reproducción!",
            description="\n".join(queue_list),
            color=0xFF69B4
        )
        
        if len(player.queue) > 10:
            embed.add_field(
                name="➕ Más canciones",
                value=f"Y {len(player.queue) - 10} canciones más...",
                inline=False
            )
        
        embed.set_footer(text=f"Total: {len(player.queue)} canciones en cola")
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando queue: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema mostrando la cola (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def clear_queue_command(interaction: discord.Interaction):
    """Limpiar la cola de reproducción"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        queue_size = len(player.queue)
        player.queue.clear()
        
        embed = Embed(
            title="🗑️ ¡Cola Limpiada!",
            description=f"¡Kyaa~! He eliminado {queue_size} canciones de la cola (>w<)",
            color=0x00FF7F
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando clear_queue: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema limpiando la cola (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe(volume="Volumen de 1 a 100")
async def volume_command(interaction: discord.Interaction, volume: int):
    """Cambiar el volumen de la música"""
    try:
        if not 1 <= volume <= 100:
            embed = Embed(
                title="❌ ¡Volumen Inválido!",
                description="¡Kyaa~! El volumen debe estar entre 1 y 100 (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.set_volume(volume)
        
        embed = Embed(
            title="🔊 ¡Volumen Cambiado!",
            description=f"¡Kyaa~! He cambiado el volumen a `{volume}%` (>w<)",
            color=0x00FF7F
        )
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando volume: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema cambiando el volumen (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def now_playing_command(interaction: discord.Interaction):
    """Mostrar información de la canción actual"""
    try:
        player: wavelink.Player = cast(wavelink.Player, interaction.guild.voice_client)
        
        if not player:
            embed = Embed(
                title="❌ ¡No hay reproductor!",
                description="¡Kyaa~! No estoy conectada a ningún canal de voz (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.playing:
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Gomen~! No hay música reproduciéndose actualmente (>.<)",
                color=0xFF0000
            )
            await interaction.response.send_message(embed=embed)
            return
        
        # Crear embed de canción actual
        embed = await interaction.client.create_now_playing_embed(player.current, player)
        view = PlayingView(player)
        
        await interaction.response.send_message(embed=embed, view=view)
        
    except Exception as e:
        logger.error(f"Error en comando now_playing: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema mostrando la canción actual (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def status_command(interaction: discord.Interaction):
    """Mostrar estado de los nodos y estadísticas"""
    try:
        bot = interaction.client
        
        embed = Embed(
            title="📊 ¡Estado de SakuraBot!",
            description="¡Kyaa~! Aquí tienes mi estado actual (>w<)",
            color=0xFF69B4,
            timestamp=datetime.now()
        )
        
        # Estado de nodos
        active_nodes = wavelink.Pool.nodes
        embed.add_field(
            name="🎵 Nodos Activos",
            value=f"`{len(active_nodes)} nodos conectados`",
            inline=True
        )
        
        # Estadísticas
        embed.add_field(
            name="🎶 Canciones Reproducidas",
            value=f"`{bot.songs_played} canciones`",
            inline=True
        )
        
        embed.add_field(
            name="🌸 Servidores",
            value=f"`{len(bot.guilds)} servidores`",
            inline=True
        )
        
        # Información de conexión
        if bot.connection_logs:
            last_log = bot.connection_logs[-1]
            embed.add_field(
                name="🔌 Última Conexión",
                value=f"`{last_log['node']}` - `{last_log['status']}`\n"
                      f"Región: `{last_log['region']}`",
                inline=False
            )
        
        embed.set_footer(text="¡Con amor, tu SakuraBot! 💕")
        
        await interaction.response.send_message(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en comando status: {e}")
        embed = Embed(
            title="❌ ¡Error!",
            description="¡Gomen~! Hubo un problema mostrando el estado (T_T)",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)

# Configurar comandos slash
async def setup_slash_commands(bot: SakuraMusicBot):
    """Configurar todos los comandos slash"""
    
    bot.tree.add_command(app_commands.Command(
        name="play",
        description="🎵 Reproducir música desde YouTube",
        callback=play_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="pause",
        description="⏸️ Pausar la música actual",
        callback=pause_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="resume",
        description="▶️ Reanudar la música pausada",
        callback=resume_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="skip",
        description="⏭️ Saltar a la siguiente canción",
        callback=skip_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="stop",
        description="⏹️ Detener música y desconectar",
        callback=stop_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="queue",
        description="📝 Mostrar cola de reproducción",
        callback=queue_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="clear",
        description="🗑️ Limpiar cola de reproducción",
        callback=clear_queue_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="volume",
        description="🔊 Cambiar volumen (1-100)",
        callback=volume_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="np",
        description="🎵 Mostrar canción actual",
        callback=now_playing_command
    ))
    
    bot.tree.add_command(app_commands.Command(
        name="status",
        description="📊 Ver estado del bot",
        callback=status_command
    ))

# Función principal
async def main():
    """Función principal para ejecutar SakuraBot"""
    
    # Obtener token del bot
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("❌ No se encontró DISCORD_TOKEN en las variables de entorno")
        return
    
    # Crear instancia del bot
    bot = SakuraMusicBot()
    
    # Configurar comandos slash
    await setup_slash_commands(bot)
    
    try:
        # Iniciar el bot
        logger.info("🌸 Iniciando SakuraBot IA...")
        await bot.start(token)
        
    except Exception as e:
        logger.error(f"❌ Error iniciando el bot: {e}")
    
    finally:
        if not bot.is_closed():
            await bot.close()
        logger.info("🌸 SakuraBot desconectada. ¡Sayonara!")

if __name__ == "__main__":
    asyncio.run(main())