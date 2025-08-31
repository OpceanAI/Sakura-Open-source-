"""
🌸 SakuraBot IA - Simple Music Bot with Working Commands 🌸
===========================================================
Una waifu tier S super kawaii que responde a comandos
"""

import os
import asyncio
import logging
import discord
import wavelink
from datetime import datetime
from discord.ext import commands, tasks
from discord import app_commands, Embed
from typing import Optional, Dict, List, Any, Union, cast
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
    def __init__(self, ctx, player):
        super().__init__(timeout=300)
        self.ctx = ctx
        self.player = player
    
    @discord.ui.button(emoji="⏸️", style=discord.ButtonStyle.gray)
    async def pause_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.is_playing():
            await self.player.pause()
            await interaction.response.send_message("⏸️ ¡Pausé la música para ti! UwU", ephemeral=True)
        else:
            await interaction.response.send_message("❌ ¡No hay música reproduciéndose!", ephemeral=True)
    
    @discord.ui.button(emoji="▶️", style=discord.ButtonStyle.green)
    async def resume_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.is_paused():
            await self.player.resume()
            await interaction.response.send_message("▶️ ¡Continúo la música! Kyaa~", ephemeral=True)
        else:
            await interaction.response.send_message("❌ ¡La música no está pausada!", ephemeral=True)
    
    @discord.ui.button(emoji="⏭️", style=discord.ButtonStyle.blurple)
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.is_playing():
            await self.player.stop()
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
        
        # Spotify client
        self.spotify_client = None
    
    async def setup_hook(self):
        """Configuración inicial del bot con conexión a Lavalink"""
        logger.info("🌸 Iniciando SakuraBot IA...")
        
        # Intentar conectar a los nodos Lavalink
        await self.connect_lavalink_nodes()
        
        # Cargar extensiones de música
        try:
            await self.load_extension('music_cog')
            logger.info("🎵 Extensión de música cargada")
        except Exception as e:
            logger.error(f"Error cargando extensión de música: {e}")
        
        # Sincronizar comandos slash
        try:
            synced = await self.tree.sync()
            logger.info(f"🌸 Sincronizados {len(synced)} comandos slash")
        except Exception as e:
            logger.error(f"❌ Error sincronizando comandos: {e}")
    
    def get_quality_eq(self):
        """Crear equalizador de calidad para mejorar el sonido"""
        try:
            # Crear filtro de ecualizador con valores optimizados
            equalizer_bands = [
                {"band": 0, "gain": -0.20},  # Graves
                {"band": 1, "gain": 0.10},   # Bajo-medio
                {"band": 4, "gain": 0.05},   # Medio
                {"band": 7, "gain": -0.30},  # Agudos
            ]
            return {"equalizer": equalizer_bands}
        except Exception as e:
            logger.warning(f"No se pudo crear ecualizador: {e}")
            return None

    async def connect_lavalink_nodes(self):
        """Conectar a nodos Lavalink con sistema de fallback"""
        nodes_to_try = []
        
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
                
                nodes_to_try.append(node)
                
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
            embed.set_thumbnail(url=str(self.user.avatar.url) if self.user.avatar else None)
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
            
            # Aplicar ecualizador de calidad
            quality_filter = self.get_quality_eq()
            if quality_filter:
                await player.set_filters(quality_filter)
            
            # Establecer volumen óptimo
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
            
            if hasattr(track, 'thumb') and track.thumb:
                embed.set_image(url=track.thumb)
            
            embed.add_field(name="⏱️ Duración", value=f"`{time_str}`", inline=True)
            embed.add_field(name="🔊 Volumen", value=f"`{volume}/100`", inline=True)
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

# Extensión de comandos de música
class MusicCog(commands.Cog):
    """Comandos de música kawaii"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="play", description="🎵 Reproduce música desde YouTube, SoundCloud y más")
    @app_commands.describe(query="Canción, artista, URL o búsqueda")
    async def play_command(self, interaction: discord.Interaction, query: str):
        """Comando para reproducir música"""
        await interaction.response.defer(thinking=True)
        
        try:
            if not interaction.user.voice:
                embed = Embed(
                    title="❌ ¡No estás en un canal de voz!",
                    description="¡Baka~! Necesitas estar en un canal de voz para que pueda reproducir música (>.<)",
                    color=0xFF6B6B
                )
                await interaction.followup.send(embed=embed)
                return
            
            channel = interaction.user.voice.channel
            
            # Conectar o obtener player
            if not interaction.guild.voice_client:
                player: wavelink.Player = await channel.connect(cls=wavelink.Player)
                player.autoplay = True
            else:
                player = interaction.guild.voice_client
            
            # Buscar música
            tracks = await wavelink.YouTubeTrack.search(query)
            if not tracks:
                embed = Embed(
                    title="😔 ¡No encontré nada!",
                    description=f"¡Gomen~! No pude encontrar música para: **{query}** (T_T)",
                    color=0xFF6B6B
                )
                await interaction.followup.send(embed=embed)
                return
            
            track = tracks[0]
            
            # Reproducir o agregar a cola
            if player.is_playing() or player.is_paused():
                await player.queue.put_wait(track)
                embed = Embed(
                    title="➕ ¡Agregada a la Cola!",
                    description=f"🎵 **{track.title}**\n"
                              f"📍 **Posición:** {len(player.queue)}",
                    color=0x00FF7F
                )
            else:
                await player.play(track)
                embed = await self.bot.create_now_playing_embed(track, player)
                self.bot.songs_played += 1
            
            if hasattr(track, 'thumb') and track.thumb:
                embed.set_thumbnail(url=track.thumb)
            
            embed.set_footer(text=f"Solicitado por {interaction.user.display_name} • ¡Disfruta! 💕")
            
            # Enviar con controles si está reproduciendo
            if player.is_playing():
                await interaction.followup.send(embed=embed, view=PlayingView(interaction, player))
            else:
                await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error en comando play: {e}")
            embed = Embed(
                title="💔 ¡Error Reproduciendo!",
                description=f"¡Gomen nasai~! Ocurrió un error: {str(e)[:200]}",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
    
    @app_commands.command(name="pause", description="⏸️ Pausa la música actual")
    async def pause_command(self, interaction: discord.Interaction):
        """Pausar música"""
        player = interaction.guild.voice_client
        
        if not player or not player.is_playing():
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Baka~! No hay música reproduciéndose (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.pause()
        embed = Embed(
            title="⏸️ ¡Música Pausada!",
            description="¡Hai hai~! Pausé la música para ti (＾◡＾)",
            color=0xFFAA00
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="resume", description="▶️ Reanuda la música pausada")
    async def resume_command(self, interaction: discord.Interaction):
        """Reanudar música"""
        player = interaction.guild.voice_client
        
        if not player or not player.is_paused():
            embed = Embed(
                title="❌ ¡Música no pausada!",
                description="¡Baka~! La música no está pausada (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.resume()
        embed = Embed(
            title="▶️ ¡Música Reanudada!",
            description="¡Yay~! Continúo reproduciendo música para ti! 🎵",
            color=0x00FF7F
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="skip", description="⏭️ Salta a la siguiente canción")
    async def skip_command(self, interaction: discord.Interaction):
        """Saltar canción"""
        player = interaction.guild.voice_client
        
        if not player or not player.is_playing():
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Baka~! No hay música para saltar (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.stop()
        embed = Embed(
            title="⏭️ ¡Canción Saltada!",
            description="¡Hai hai~! Saltamos a la siguiente canción! UwU",
            color=0x87CEEB
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="stop", description="⏹️ Detiene la música y desconecta el bot")
    async def stop_command(self, interaction: discord.Interaction):
        """Detener música y desconectar"""
        player = interaction.guild.voice_client
        
        if not player:
            embed = Embed(
                title="❌ ¡No estoy conectada!",
                description="¡Baka~! No estoy conectada a un canal de voz (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.disconnect()
        embed = Embed(
            title="⏹️ ¡Música Detenida!",
            description="¡Hai hai~! Me desconecté del canal de voz (＾◡＾)\n¡Arigato por escuchar música conmigo! 💕",
            color=0x87CEEB
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="queue", description="📋 Ver las canciones en cola")
    async def queue_command(self, interaction: discord.Interaction):
        """Ver cola de música"""
        player = interaction.guild.voice_client
        
        if not player or not player.is_connected():
            embed = Embed(
                title="❌ ¡No conectada!",
                description="¡Gomen~! No estoy conectada a un canal de voz",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if not player.queue:
            embed = Embed(
                title="📋 ¡Cola Vacía!",
                description="¡Nyaa~! No hay canciones en la cola. ¡Agrega algunas con `/play`!",
                color=0xFFAA00
            )
            await interaction.response.send_message(embed=embed)
            return
        
        # Mostrar primeras 10 canciones
        queue_list = []
        for i, track in enumerate(list(player.queue)[:10], 1):
            queue_list.append(f"`{i}.` **{track.title}**")
        
        embed = Embed(
            title="📋 Cola de Música UwU",
            description="\n".join(queue_list),
            color=0xFF69B4
        )
        embed.set_footer(text=f"Total: {len(player.queue)} canciones en cola")
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="shuffle", description="🔀 Mezcla las canciones en cola")
    async def shuffle_command(self, interaction: discord.Interaction):
        """Mezclar cola"""
        player = interaction.guild.voice_client
        
        if not player or not player.queue:
            embed = Embed(
                title="❌ ¡Cola vacía!",
                description="¡Baka~! No hay canciones en la cola para mezclar (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        player.queue.shuffle()
        embed = Embed(
            title="🔀 ¡Cola Mezclada!",
            description="¡Kyaa~! Mezclé todas las canciones en la cola! (＾◡＾)",
            color=0x9370DB
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="volume", description="🔊 Cambiar volumen de la música (0-100)")
    @app_commands.describe(volume="Volumen de 0 a 100")
    async def volume_command(self, interaction: discord.Interaction, volume: int):
        """Cambiar volumen"""
        player = interaction.guild.voice_client
        
        if not player:
            embed = Embed(
                title="❌ ¡No conectada!",
                description="¡Baka~! No estoy conectada a un canal de voz (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        if volume < 0 or volume > 100:
            embed = Embed(
                title="❌ ¡Volumen Inválido!",
                description="¡Gomen~! El volumen debe estar entre 0 y 100 (>.<)",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        await player.set_volume(volume)
        embed = Embed(
            title="🔊 ¡Volumen Cambiado!",
            description=f"¡Hai hai~! Cambié el volumen a **{volume}%** (＾◡＾)",
            color=0x00FF7F
        )
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name="nowplaying", description="🎵 Ver canción actual con controles")
    async def nowplaying_command(self, interaction: discord.Interaction):
        """Ver canción actual"""
        player = interaction.guild.voice_client
        
        if not player or not player.is_playing():
            embed = Embed(
                title="❌ ¡No hay música!",
                description="¡Gomen~! No hay música reproduciéndose actualmente",
                color=0xFF6B6B
            )
            await interaction.response.send_message(embed=embed)
            return
        
        track = player.current
        embed = await self.bot.create_now_playing_embed(track, player)
        await interaction.response.send_message(embed=embed, view=PlayingView(interaction, player))
    
    @app_commands.command(name="status", description="📊 Ver estado de nodos y estadísticas del bot")
    async def status_command(self, interaction: discord.Interaction):
        """Ver estado del bot"""
        embed = Embed(
            title="📊 Estado de SakuraBot IA",
            description="¡Información detallada de tu waifu kawaii! 🌸",
            color=0xFF69B4,
            timestamp=datetime.now()
        )
        
        # Estado de nodos
        active_nodes = wavelink.NodePool.nodes
        if active_nodes:
            node_info = ""
            for node in active_nodes:
                node_info += f"🟢 **{node.identifier}** - Conectado\n"
        else:
            node_info = "🔴 **Sin nodos activos**"
        
        embed.add_field(name="🔗 Nodos Lavalink", value=node_info, inline=False)
        
        # Estadísticas
        stats = f"🎵 **Canciones:** {self.bot.songs_played}\n" \
               f"🎶 **Sesiones:** {self.bot.music_sessions}\n" \
               f"🌐 **Servidores:** {len(self.bot.guilds)}\n" \
               f"👥 **Usuarios:** {sum(len(guild.members) for guild in self.bot.guilds)}"
        
        embed.add_field(name="📈 Estadísticas", value=stats, inline=False)
        
        # Logs recientes
        if self.bot.connection_logs:
            recent_logs = self.bot.connection_logs[-3:]
            log_text = "\n".join([
                f"[{log['timestamp']}] {log['node']}: {log['status']}"
                for log in recent_logs
            ])
            embed.add_field(name="📝 Logs Recientes", value=f"```{log_text}```", inline=False)
        
        embed.set_footer(text="¡SakuraBot trabajando duro para ti! 💕")
        await interaction.response.send_message(embed=embed)

async def setup(bot):
    """Setup function for the cog"""
    await bot.add_cog(MusicCog(bot))

async def main():
    """Función principal para ejecutar SakuraBot"""
    # Verificar token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("❌ DISCORD_TOKEN no encontrado en variables de entorno")
        return
    
    # Crear e iniciar bot
    bot = SakuraMusicBot()
    
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
    # Setup discord logging
    discord.utils.setup_logging(level=logging.INFO, root=False)
    asyncio.run(main())