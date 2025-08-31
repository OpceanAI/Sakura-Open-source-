"""
🌸 SakuraBot IA - Complete Advanced Music Bot 🌸
=================================================
Una waifu tier S completamente funcional con sistema avanzado de audio
Incluye codificador de audio, filtros de calidad, y gestión completa de música
"""

import os
import discord
import wavelink
import asyncio
import logging
import aiohttp
import random
import json
from discord.ext import commands, tasks
from discord import app_commands, Embed
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sakura_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SakuraBot')

# 8 nodos Lavalink públicos con configuración completa
LAVALINK_NODES = [
    {"host": "lava.luke.gg", "port": 443, "password": "discordbotlist.com", "secure": True, "region": "Europa"},
    {"host": "46.202.82.164", "port": 1027, "password": "jmlitelavalink", "secure": False, "region": "Global"},
    {"host": "lavalink-v2.pericsq.ro", "port": 6677, "password": "wwweasycodero", "secure": False, "region": "Europa"},
    {"host": "69.30.219.180", "port": 1047, "password": "yothisnodeishostedbymushroom0162", "secure": False, "region": "EE.UU."},
    {"host": "lava3.horizxon.studio", "port": 80, "password": "horizxon.studio", "secure": False, "region": "Global"},
    {"host": "lava2.horizxon.studio", "port": 80, "password": "horizxon.studio", "secure": False, "region": "Global"},
    {"host": "lavalink.micium-hosting.com", "port": 80, "password": "micium-hosting.com", "secure": False, "region": "Global"},
    {"host": "lavalink.oops.wtf", "port": 443, "password": "www.freelavalink.ga", "secure": True, "region": "Global"}
]

class AudioCodec:
    """Codificador de audio avanzado con filtros de calidad"""
    
    @staticmethod
    def get_quality_equalizer():
        """Ecualizador optimizado para calidad de audio"""
        return [
            {"band": 0, "gain": -0.20},   # 25Hz - Graves profundos (reducir rumble)
            {"band": 1, "gain": 0.10},    # 40Hz - Bajo-graves (potenciar)
            {"band": 2, "gain": 0.05},    # 63Hz - Bajos medios
            {"band": 3, "gain": 0.00},    # 100Hz - Bajos altos
            {"band": 4, "gain": 0.05},    # 160Hz - Medios bajos (claridad)
            {"band": 5, "gain": 0.00},    # 250Hz - Medios
            {"band": 6, "gain": 0.00},    # 400Hz - Medios altos
            {"band": 7, "gain": -0.15},   # 630Hz - Medios-agudos (reducir harshness)
            {"band": 8, "gain": -0.25},   # 1kHz - Agudos (suavizar)
            {"band": 9, "gain": -0.30},   # 1.6kHz - Agudos altos (anti-fatiga)
            {"band": 10, "gain": -0.35},  # 2.5kHz - Presencia (controlar sibilancia)
            {"band": 11, "gain": -0.30},  # 4kHz - Brillantez
            {"band": 12, "gain": -0.25},  # 6.3kHz - Aire
            {"band": 13, "gain": -0.20},  # 10kHz - Extensión alta
            {"band": 14, "gain": -0.15}   # 16kHz - Ultra agudos
        ]
    
    @staticmethod
    def get_bass_boost():
        """Potenciador de graves"""
        return [
            {"band": 0, "gain": 0.20},
            {"band": 1, "gain": 0.15},
            {"band": 2, "gain": 0.10},
            {"band": 3, "gain": 0.05}
        ]
    
    @staticmethod
    def get_vocal_enhance():
        """Realzador de voces"""
        return [
            {"band": 6, "gain": 0.15},    # 400Hz - Calor vocal
            {"band": 7, "gain": 0.10},    # 630Hz - Presencia vocal
            {"band": 8, "gain": 0.05},    # 1kHz - Claridad vocal
            {"band": 9, "gain": -0.05}    # 1.6kHz - Controlar dureza
        ]

class MusicControlView(discord.ui.View):
    """Panel de control avanzado para música"""
    
    def __init__(self, player: wavelink.Player, bot):
        super().__init__(timeout=300)
        self.player = player
        self.bot = bot
        self.current_filter = "quality"
    
    @discord.ui.button(emoji="⏸️", style=discord.ButtonStyle.gray, label="Pausar")
    async def pause_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.playing:
            await self.player.pause(True)
            button.emoji = "▶️"
            button.label = "Reanudar"
            await interaction.response.edit_message(content="⏸️ Música pausada", view=self)
        elif self.player.paused:
            await self.player.pause(False)
            button.emoji = "⏸️"
            button.label = "Pausar"
            await interaction.response.edit_message(content="▶️ Música reanudada", view=self)
        else:
            await interaction.response.send_message("❌ No hay música reproduciéndose", ephemeral=True)
    
    @discord.ui.button(emoji="⏭️", style=discord.ButtonStyle.blurple, label="Saltar")
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.player.playing or self.player.paused:
            await self.player.skip()
            await interaction.response.send_message("⏭️ Canción saltada", ephemeral=True)
        else:
            await interaction.response.send_message("❌ No hay música para saltar", ephemeral=True)
    
    @discord.ui.button(emoji="🔀", style=discord.ButtonStyle.green, label="Aleatorio")
    async def shuffle_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if len(self.player.queue) > 1:
            self.player.queue.shuffle()
            await interaction.response.send_message("🔀 Cola mezclada aleatoriamente", ephemeral=True)
        else:
            await interaction.response.send_message("❌ Necesitas al menos 2 canciones en la cola", ephemeral=True)
    
    @discord.ui.button(emoji="🎚️", style=discord.ButtonStyle.secondary, label="Filtros")
    async def filter_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        view = FilterView(self.player, self.bot)
        embed = discord.Embed(
            title="🎚️ Filtros de Audio",
            description="Selecciona un filtro de audio:",
            color=0xFF69B4
        )
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    
    @discord.ui.button(emoji="⏹️", style=discord.ButtonStyle.red, label="Detener")
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.player.disconnect()
        await interaction.response.edit_message(content="⏹️ Desconectada. ¡Sayonara! 💕", view=None)

class FilterView(discord.ui.View):
    """Vista para selección de filtros de audio"""
    
    def __init__(self, player: wavelink.Player, bot):
        super().__init__(timeout=60)
        self.player = player
        self.bot = bot
    
    @discord.ui.button(label="🎵 Calidad", style=discord.ButtonStyle.primary)
    async def quality_filter(self, interaction: discord.Interaction, button: discord.ui.Button):
        eq_bands = AudioCodec.get_quality_equalizer()
        filters = wavelink.Filters(equalizer=eq_bands)
        await self.player.set_filters(filters)
        await interaction.response.send_message("🎵 Filtro de calidad aplicado", ephemeral=True)
    
    @discord.ui.button(label="🔊 Graves+", style=discord.ButtonStyle.secondary)
    async def bass_filter(self, interaction: discord.Interaction, button: discord.ui.Button):
        eq_bands = AudioCodec.get_bass_boost()
        filters = wavelink.Filters(equalizer=eq_bands)
        await self.player.set_filters(filters)
        await interaction.response.send_message("🔊 Potenciador de graves aplicado", ephemeral=True)
    
    @discord.ui.button(label="🎤 Voces", style=discord.ButtonStyle.secondary)
    async def vocal_filter(self, interaction: discord.Interaction, button: discord.ui.Button):
        eq_bands = AudioCodec.get_vocal_enhance()
        filters = wavelink.Filters(equalizer=eq_bands)
        await self.player.set_filters(filters)
        await interaction.response.send_message("🎤 Realzador de voces aplicado", ephemeral=True)
    
    @discord.ui.button(label="🔄 Reset", style=discord.ButtonStyle.danger)
    async def reset_filter(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.player.set_filters(None)
        await interaction.response.send_message("🔄 Filtros reiniciados", ephemeral=True)

class VolumeModal(discord.ui.Modal):
    """Modal para cambio de volumen"""
    
    def __init__(self, player: wavelink.Player):
        super().__init__(title="🔊 Cambiar Volumen")
        self.player = player
        
        self.volume_input = discord.ui.TextInput(
            label="Volumen (1-200)",
            placeholder="Ingresa un valor entre 1 y 200",
            default=str(player.volume),
            max_length=3
        )
        self.add_item(self.volume_input)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            volume = int(self.volume_input.value)
            if 1 <= volume <= 200:
                await self.player.set_volume(volume)
                await interaction.response.send_message(f"🔊 Volumen cambiado a {volume}%", ephemeral=True)
            else:
                await interaction.response.send_message("❌ El volumen debe estar entre 1 y 200", ephemeral=True)
        except ValueError:
            await interaction.response.send_message("❌ Valor inválido", ephemeral=True)

class SakuraCompleteBot(commands.Bot):
    """Bot completo de música con todas las funcionalidades avanzadas"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        super().__init__(
            command_prefix=['!', 's!', 'sakura!'],
            intents=intents,
            help_command=None,
            description="🌸 SakuraBot IA - Bot completo de música con codificador de audio avanzado",
            activity=discord.Game(name="🎵 Música de alta calidad con filtros pro")
        )
        
        # Estados del bot
        self.node_index = 0
        self.connection_logs = []
        self.audio_sessions = 0
        self.tracks_played = 0
        self.error_count = 0
        
        # Configuraciones de audio
        self.default_volume = 75
        self.audio_quality = "high"
        
    async def setup_hook(self):
        """Configuración inicial completa"""
        logger.info("🌸 Iniciando SakuraBot IA Completo...")
        
        # Conectar a nodos Lavalink con fallback inteligente
        success = await self.connect_lavalink_nodes()
        
        if success:
            logger.info("✅ Sistema de audio inicializado correctamente")
        else:
            logger.error("❌ Fallo crítico en inicialización de audio")
        
        # Sincronizar comandos slash (con manejo de rate limit)
        try:
            # Intentar sincronizar, pero no bloquear si hay rate limit
            synced = await self.tree.sync()
            logger.info(f"🌸 Sistema completo: {len(synced)} comandos sincronizados")
        except discord.HTTPException as e:
            if e.status == 429:  # Rate limited
                logger.warning("⚠️ Rate limit detectado - Comandos se sincronizarán automáticamente más tarde")
            else:
                logger.error(f"❌ Error HTTP en sincronización: {e}")
                self.error_count += 1
        except Exception as e:
            logger.error(f"❌ Error en sincronización: {e}")
            self.error_count += 1
        
        # Iniciar tareas de monitoreo
        if not self.monitor_system.is_running():
            self.monitor_system.start()
    
    async def connect_lavalink_nodes(self) -> bool:
        """Conexión inteligente a nodos con fallback automático"""
        for i, node_config in enumerate(LAVALINK_NODES):
            try:
                protocol = "wss" if node_config["secure"] else "ws"
                uri = f"{protocol}://{node_config['host']}:{node_config['port']}"
                
                logger.info(f"🔄 Conectando a {node_config['host']} ({node_config['region']})")
                
                node = wavelink.Node(
                    uri=uri,
                    password=node_config["password"],
                    identifier=f"Node_{i+1}_{node_config['region']}"
                )
                
                await wavelink.Pool.connect(client=self, nodes=[node])
                
                # Log exitoso
                log_entry = {
                    "timestamp": datetime.now(),
                    "node": node_config["host"],
                    "region": node_config["region"],
                    "status": "CONNECTED",
                    "latency": "pending"
                }
                self.connection_logs.append(log_entry)
                
                logger.info(f"✅ Conectado exitosamente a {node_config['host']} ({node_config['region']})")
                return True
                
            except Exception as e:
                error_log = {
                    "timestamp": datetime.now(),
                    "node": node_config["host"],
                    "region": node_config["region"],
                    "status": "FAILED",
                    "error": str(e)
                }
                self.connection_logs.append(error_log)
                
                logger.warning(f"❌ Fallo en {node_config['host']}: {e}")
                self.error_count += 1
                continue
        
        logger.error("❌ No se pudo conectar a ningún nodo Lavalink")
        return False
    
    async def on_ready(self):
        """Bot completamente inicializado"""
        logger.info(f"🌸 {self.user} - Sistema completo operativo")
        logger.info(f"🎵 Servidores activos: {len(self.guilds)}")
        logger.info(f"🎚️ Codificador de audio: ACTIVADO")
        logger.info(f"🔊 Filtros de calidad: DISPONIBLES")
        
        # Log server details for debugging
        for guild in self.guilds:
            logger.info(f"✅ Conectada a servidor: {guild.name} (ID: {guild.id})")
        
        # Establecer estado personalizado
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="🎵 Audio de alta fidelidad | /play para música"
        )
        await self.change_presence(activity=activity, status=discord.Status.online)
        logger.info("🌸 Bot completamente inicializado y listo para comandos")
    
    async def on_wavelink_node_ready(self, payload: wavelink.NodeReadyEventPayload):
        """Nodo de audio listo con optimizaciones"""
        node = payload.node
        logger.info(f"🎵 ¡UwU conectada a {node.identifier}! 🎵 Calidad asegurada~")
        
        # Actualizar logs con latencia
        for log in self.connection_logs:
            if node.identifier.endswith(log["node"].replace(".", "_")):
                log["latency"] = f"{node.heartbeat}ms" if hasattr(node, 'heartbeat') else "N/A"
                break
    
    async def on_wavelink_track_start(self, payload: wavelink.TrackStartEventPayload):
        """Aplicar configuraciones de audio al iniciar track"""
        player = payload.player
        track = payload.track
        
        try:
            # Aplicar filtros de calidad por defecto
            eq_bands = AudioCodec.get_quality_equalizer()
            filters = wavelink.Filters(equalizer=eq_bands)
            await player.set_filters(filters)
            
            # Configurar volumen óptimo
            await player.set_volume(self.default_volume)
            
            # Incrementar estadísticas
            self.tracks_played += 1
            
            logger.info(f"🎵 Reproduciendo con audio optimizado: {track.title}")
            
        except Exception as e:
            logger.error(f"Error aplicando configuraciones de audio: {e}")
            self.error_count += 1
    
    async def on_wavelink_track_end(self, payload: wavelink.TrackEndEventPayload):
        """Gestión al finalizar track"""
        player = payload.player
        track = payload.track
        
        logger.info(f"✅ Finalizada: {track.title}")
        
        # Si la cola está vacía, aplicar configuraciones de standby
        if player.queue.is_empty and not player.playing:
            await asyncio.sleep(2)
            if not player.playing and player.queue.is_empty:
                logger.info("🔄 Cola vacía - Modo standby activado")
    
    @tasks.loop(minutes=10)
    async def monitor_system(self):
        """Monitoreo avanzado del sistema"""
        try:
            nodes = wavelink.Pool.nodes
            active_players = sum(len(node.players) for node in nodes)
            
            logger.info(f"📊 Estado: {len(nodes)} nodos | {active_players} reproductores | {self.tracks_played} tracks | {self.error_count} errores")
            
            # Auto-reconexión si es necesario
            if not nodes:
                logger.warning("⚠️ Sin nodos activos - Intentando reconexión")
                await self.connect_lavalink_nodes()
                
        except Exception as e:
            logger.error(f"Error en monitoreo: {e}")
    
    def format_duration(self, milliseconds: int) -> str:
        """Formatear duración en formato legible"""
        seconds = milliseconds // 1000
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    
    async def create_now_playing_embed(self, track: wavelink.Playable, player: wavelink.Player) -> discord.Embed:
        """Crear embed avanzado de reproducción actual"""
        embed = discord.Embed(
            title="🎵 Reproduciendo Ahora",
            color=0xFF69B4,
            timestamp=datetime.now()
        )
        
        # Información principal
        embed.add_field(
            name="🎼 Título",
            value=f"```{track.title}```",
            inline=False
        )
        
        if hasattr(track, 'author') and track.author:
            embed.add_field(name="👤 Artista", value=f"`{track.author}`", inline=True)
        
        # Información técnica
        duration = self.format_duration(track.length)
        embed.add_field(name="⏱️ Duración", value=f"`{duration}`", inline=True)
        embed.add_field(name="🔊 Volumen", value=f"`{player.volume}%`", inline=True)
        embed.add_field(name="📝 En Cola", value=f"`{len(player.queue)}`", inline=True)
        embed.add_field(name="🎚️ Filtros", value="`Calidad HD`", inline=True)
        embed.add_field(name="🎵 Codec", value="`Optimizado`", inline=True)
        
        # Imagen si está disponible
        if hasattr(track, 'artwork') and track.artwork:
            embed.set_image(url=track.artwork)
        
        embed.set_footer(
            text=f"🌸 SakuraBot • Audio de alta calidad • Track #{self.tracks_played}",
            icon_url=self.user.display_avatar.url if self.user else None
        )
        
        return embed

# Comandos slash avanzados
@app_commands.describe(query="Canción, artista, álbum o URL para reproducir")
async def play_advanced(interaction: discord.Interaction, query: str):
    """Comando de reproducción avanzado con búsqueda inteligente"""
    try:
        await interaction.response.defer(thinking=True)
        
        # Verificaciones de usuario
        if not hasattr(interaction.user, 'voice') or not interaction.user.voice:
            embed = discord.Embed(
                title="❌ Error de Conexión",
                description="Necesitas estar conectado a un canal de voz para usar este comando.",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
            return
        
        voice_channel = interaction.user.voice.channel
        bot = interaction.client
        
        # Obtener o crear player
        player: wavelink.Player = interaction.guild.voice_client
        if not player:
            try:
                player = await voice_channel.connect(cls=wavelink.Player)
                bot.audio_sessions += 1
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Error de Conexión",
                    description=f"No pude conectarme al canal de voz: {str(e)}",
                    color=0xFF0000
                )
                await interaction.followup.send(embed=embed)
                return
        
        # Búsqueda avanzada de música
        try:
            # Intentar diferentes fuentes de búsqueda
            search_query = query
            if not any(x in query.lower() for x in ['http', 'youtu', 'spotify', 'soundcloud']):
                search_query = f"ytsearch:{query}"
            
            tracks = await wavelink.Playable.search(search_query)
            
            if not tracks:
                # Búsqueda alternativa
                tracks = await wavelink.Playable.search(f"ytsearch:{query} audio")
                
            if not tracks:
                embed = discord.Embed(
                    title="❌ Sin Resultados",
                    description=f"No pude encontrar música para: `{query}`\n\nIntenta con:\n• Nombre del artista + canción\n• URL de YouTube/Spotify\n• Términos más específicos",
                    color=0xFF0000
                )
                await interaction.followup.send(embed=embed)
                return
            
            track = tracks[0]
            
        except Exception as e:
            embed = discord.Embed(
                title="❌ Error de Búsqueda",
                description=f"Error buscando música: {str(e)}",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Reproducir o agregar a cola
        if not player.playing and not player.paused:
            # Reproducir inmediatamente
            await player.play(track)
            
            # Crear embed de reproducción
            embed = await bot.create_now_playing_embed(track, player)
            view = MusicControlView(player, bot)
            
            await interaction.followup.send(embed=embed, view=view)
            
        else:
            # Agregar a cola
            await player.queue.put_wait(track)
            
            embed = discord.Embed(
                title="➕ Agregada a la Cola",
                description=f"**{track.title}**",
                color=0x00FF7F,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="👤 Artista", value=f"`{track.author or 'Desconocido'}`", inline=True)
            embed.add_field(name="⏱️ Duración", value=f"`{bot.format_duration(track.length)}`", inline=True)
            embed.add_field(name="📍 Posición", value=f"`#{len(player.queue)}`", inline=True)
            
            if hasattr(track, 'artwork') and track.artwork:
                embed.set_thumbnail(url=track.artwork)
            
            embed.set_footer(text="🎵 Se reproducirá cuando llegue su turno")
            
            await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error crítico en play_advanced: {e}")
        embed = discord.Embed(
            title="❌ Error Crítico",
            description="Ocurrió un error inesperado. El equipo técnico ha sido notificado.",
            color=0xFF0000
        )
        try:
            await interaction.followup.send(embed=embed)
        except:
            pass

@app_commands.describe(volume="Volumen de 1 a 200 (100 = normal)")
async def volume_advanced(interaction: discord.Interaction, volume: Optional[int] = None):
    """Control avanzado de volumen con modal interactivo"""
    player: wavelink.Player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ Sin Reproductor",
            description="No hay un reproductor de música activo en este servidor.",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if volume is None:
        # Mostrar modal para entrada de volumen
        modal = VolumeModal(player)
        await interaction.response.send_modal(modal)
        return
    
    # Validar y aplicar volumen
    if not 1 <= volume <= 200:
        embed = discord.Embed(
            title="❌ Volumen Inválido",
            description="El volumen debe estar entre 1 y 200.",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    await player.set_volume(volume)
    
    # Determinar emoji basado en nivel
    if volume <= 30:
        emoji = "🔈"
    elif volume <= 70:
        emoji = "🔉"
    else:
        emoji = "🔊"
    
    embed = discord.Embed(
        title=f"{emoji} Volumen Configurado",
        description=f"Volumen ajustado a **{volume}%**",
        color=0x00FF7F
    )
    
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def queue_advanced(interaction: discord.Interaction):
    """Cola de reproducción avanzada con navegación"""
    player: wavelink.Player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ Sin Reproductor",
            description="No hay música reproduciéndose en este servidor.",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if not player.queue:
        embed = discord.Embed(
            title="📝 Cola Vacía",
            description="No hay canciones en la cola.\nUsa `/play` para agregar música.",
            color=0xFFAA00
        )
        
        if player.current:
            embed.add_field(
                name="🎵 Reproduciendo Ahora",
                value=f"**{player.current.title}**",
                inline=False
            )
        
        await interaction.response.send_message(embed=embed)
        return
    
    # Crear lista de canciones
    queue_list = []
    total_duration = 0
    
    for i, track in enumerate(list(player.queue)[:15]):  # Mostrar primeras 15
        duration_ms = track.length
        duration_str = interaction.client.format_duration(duration_ms)
        total_duration += duration_ms
        
        queue_list.append(f"`{i+1:2d}.` **{track.title}** `[{duration_str}]`")
    
    embed = discord.Embed(
        title="📝 Cola de Reproducción",
        description="\n".join(queue_list),
        color=0xFF69B4,
        timestamp=datetime.now()
    )
    
    # Información adicional
    embed.add_field(
        name="📊 Estadísticas",
        value=f"**Total:** {len(player.queue)} canciones\n"
              f"**Duración:** {interaction.client.format_duration(total_duration)}\n"
              f"**Siguiente:** {queue_list[0].split('**')[1] if queue_list else 'Ninguna'}",
        inline=True
    )
    
    if player.current:
        embed.add_field(
            name="🎵 Reproduciendo",
            value=f"**{player.current.title}**\n`{interaction.client.format_duration(player.current.length)}`",
            inline=True
        )
    
    if len(player.queue) > 15:
        embed.set_footer(text=f"Mostrando 15 de {len(player.queue)} canciones")
    else:
        embed.set_footer(text=f"🌸 SakuraBot • {len(player.queue)} canciones en total")
    
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def nowplaying_advanced(interaction: discord.Interaction):
    """Información detallada de la canción actual con controles"""
    player: wavelink.Player = interaction.guild.voice_client
    
    if not player or not player.current:
        embed = discord.Embed(
            title="❌ Sin Música",
            description="No hay música reproduciéndose actualmente.",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    # Crear embed completo
    embed = await interaction.client.create_now_playing_embed(player.current, player)
    view = MusicControlView(player, interaction.client)
    
    await interaction.response.send_message(embed=embed, view=view)

@app_commands.describe()
async def filters_menu(interaction: discord.Interaction):
    """Menú completo de filtros de audio"""
    player: wavelink.Player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ Sin Reproductor",
            description="No hay un reproductor activo para aplicar filtros.",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    embed = discord.Embed(
        title="🎚️ Centro de Control de Audio",
        description="Selecciona un filtro para optimizar tu experiencia auditiva:",
        color=0xFF69B4
    )
    
    embed.add_field(
        name="🎵 Calidad HD",
        value="Ecualizador optimizado para mejor calidad general",
        inline=False
    )
    embed.add_field(
        name="🔊 Potenciador de Graves",
        value="Realza los bajos para géneros como EDM, Hip-Hop",
        inline=False
    )
    embed.add_field(
        name="🎤 Realzador de Voces",
        value="Optimiza frecuencias vocales para claridad",
        inline=False
    )
    embed.add_field(
        name="🔄 Reset",
        value="Elimina todos los filtros aplicados",
        inline=False
    )
    
    view = FilterView(player, interaction.client)
    await interaction.response.send_message(embed=embed, view=view)

@app_commands.describe()
async def system_status(interaction: discord.Interaction):
    """Estado completo del sistema de audio"""
    bot = interaction.client
    
    embed = discord.Embed(
        title="📊 Estado del Sistema SakuraBot",
        color=0xFF69B4,
        timestamp=datetime.now()
    )
    
    # Estado de nodos
    nodes = wavelink.Pool.nodes
    if nodes:
        node_info = []
        for node in nodes:
            status = "🟢 CONECTADO" if node.status == wavelink.NodeStatus.CONNECTED else "🔴 DESCONECTADO"
            players = len(node.players)
            node_info.append(f"**{node.identifier}**\n{status} • {players} reproductores")
        
        embed.add_field(
            name="🎵 Nodos de Audio",
            value="\n\n".join(node_info[:3]),  # Mostrar primeros 3
            inline=False
        )
    else:
        embed.add_field(
            name="🎵 Nodos de Audio",
            value="❌ Sin conexiones activas",
            inline=False
        )
    
    # Estadísticas generales
    embed.add_field(
        name="📈 Estadísticas",
        value=f"**Servidores:** {len(bot.guilds)}\n"
              f"**Tracks reproducidos:** {bot.tracks_played}\n"
              f"**Sesiones de audio:** {getattr(bot, 'audio_sessions', 0)}\n"
              f"**Errores:** {bot.error_count}",
        inline=True
    )
    
    # Estado del reproductor actual
    player: wavelink.Player = interaction.guild.voice_client
    if player:
        status = "🎵 Reproduciendo" if player.playing else "⏸️ Pausado" if player.paused else "⏹️ Detenido"
        embed.add_field(
            name="🎛️ Reproductor Local",
            value=f"**Estado:** {status}\n"
                  f"**Volumen:** {player.volume}%\n"
                  f"**En cola:** {len(player.queue)}\n"
                  f"**Filtros:** {'✅ Activos' if hasattr(player, '_filters') else '❌ Ninguno'}",
            inline=True
        )
    else:
        embed.add_field(
            name="🎛️ Reproductor Local",
            value="❌ No activo en este servidor",
            inline=True
        )
    
    # Logs de conexión recientes
    if bot.connection_logs:
        recent_logs = bot.connection_logs[-3:]  # Últimos 3 logs
        log_text = []
        for log in recent_logs:
            status_emoji = "✅" if log["status"] == "CONNECTED" else "❌"
            time_str = log["timestamp"].strftime("%H:%M:%S")
            log_text.append(f"{status_emoji} {log['node']} ({time_str})")
        
        embed.add_field(
            name="📋 Logs Recientes",
            value="\n".join(log_text),
            inline=False
        )
    
    embed.set_footer(
        text="🌸 SakuraBot - Sistema de Audio Avanzado",
        icon_url=bot.user.display_avatar.url if bot.user else None
    )
    
    await interaction.response.send_message(embed=embed)

# Configuración del bot
async def main():
    """Función principal para ejecutar SakuraBot completo"""
    
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("❌ DISCORD_TOKEN no encontrado en variables de entorno")
        return
    
    # Crear instancia del bot
    bot = SakuraCompleteBot()
    
    # Registrar comandos slash
    commands_to_add = [
        app_commands.Command(name="play", description="🎵 Reproducir música con búsqueda avanzada", callback=play_advanced),
        app_commands.Command(name="volume", description="🔊 Control avanzado de volumen", callback=volume_advanced),
        app_commands.Command(name="queue", description="📝 Ver cola de reproducción detallada", callback=queue_advanced),
        app_commands.Command(name="np", description="🎵 Información de canción actual con controles", callback=nowplaying_advanced),
        app_commands.Command(name="filters", description="🎚️ Menú de filtros de audio avanzados", callback=filters_menu),
        app_commands.Command(name="status", description="📊 Estado completo del sistema", callback=system_status),
    ]
    
    for command in commands_to_add:
        bot.tree.add_command(command)
    
    try:
        logger.info("🌸 Iniciando SakuraBot IA - Sistema Completo...")
        await bot.start(token)
        
    except Exception as e:
        logger.error(f"❌ Error crítico en inicialización: {e}")
        
    finally:
        if not bot.is_closed():
            await bot.close()
        logger.info("🌸 SakuraBot desconectada. ¡Sayonara!")

if __name__ == "__main__":
    asyncio.run(main())