"""
🌸 SakuraBot IA - No Rate Limit Version 🌸
==========================================
Bot completo sin sincronización de comandos para evitar rate limits
"""

import os
import discord
import wavelink
import asyncio
import logging
from discord.ext import commands
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SakuraBot')

# 8 nodos prioritizados
NODES = [
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
    """Codificador de audio con filtros de calidad"""
    
    @staticmethod
    def get_quality_filters():
        return [
            {"band": 0, "gain": -0.20},   # Graves
            {"band": 1, "gain": 0.10},    # Bajo-medio
            {"band": 4, "gain": 0.05},    # Medio
            {"band": 7, "gain": -0.30},   # Agudos
        ]

class SakuraBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        
        super().__init__(
            command_prefix=['?', 's?', 'sakura?'],
            intents=intents,
            help_command=None
        )
        
        self.tracks_played = 0
        self.connected_node = None

    async def setup_hook(self):
        """Setup sin sincronización de comandos"""
        logger.info("🌸 Iniciando SakuraBot sin rate limits...")
        
        # Conectar a nodos sin bloquear
        await self.connect_nodes()
        
        # NO sincronizar comandos para evitar rate limits
        logger.info("⚠️ Comandos slash deshabilitados para evitar rate limits")
        logger.info("✅ Usar comandos de texto: ?play, ?pause, ?resume, ?skip, ?stop")

    async def connect_nodes(self):
        """Conectar a nodos con fallback"""
        for i, node_config in enumerate(NODES):
            try:
                protocol = "wss" if node_config["secure"] else "ws"
                uri = f"{protocol}://{node_config['host']}:{node_config['port']}"
                
                logger.info(f"🔄 Conectando a {node_config['host']} ({node_config['region']})")
                
                node = wavelink.Node(
                    uri=uri,
                    password=node_config["password"],
                    identifier=f"Node{i+1}"
                )
                
                await wavelink.Pool.connect(client=self, nodes=[node])
                self.connected_node = node_config
                logger.info(f"✅ Conectado a {node_config['host']} ({node_config['region']})")
                return True
                
            except Exception as e:
                logger.warning(f"❌ Falló {node_config['host']}: {e}")
                continue
        
        logger.error("❌ No se pudo conectar a ningún nodo")
        return False

    async def on_ready(self):
        """Bot listo"""
        logger.info(f"🌸 {self.user} conectada!")
        logger.info(f"🌸 En {len(self.guilds)} servidores")
        
        for guild in self.guilds:
            logger.info(f"✅ Servidor: {guild.name} (ID: {guild.id})")
        
        logger.info("🎵 Comandos disponibles: ?play, ?pause, ?resume, ?skip, ?stop, ?volume, ?queue")
        logger.info("🌸 Bot completamente inicializado y listo")

    async def on_wavelink_node_ready(self, payload):
        """Nodo listo"""
        logger.info(f"🎵 UwU conectada a {payload.node.identifier}! Calidad asegurada~")

    async def on_wavelink_track_start(self, payload):
        """Aplicar filtros al iniciar track"""
        try:
            player = payload.player
            track = payload.track
            
            # Aplicar filtros de calidad
            eq_bands = AudioCodec.get_quality_filters()
            filters = wavelink.Filters(equalizer=eq_bands)
            await player.set_filters(filters)
            await player.set_volume(75)
            
            self.tracks_played += 1
            logger.info(f"🎵 Reproduciendo con filtros: {track.title}")
            
        except Exception as e:
            logger.error(f"Error aplicando filtros: {e}")

# Comandos de texto (sin slash commands)
@commands.command(name='play', aliases=['p'])
async def play_command(ctx, *, query):
    """Reproducir música"""
    try:
        if not ctx.author.voice:
            await ctx.send("❌ Necesitas estar en un canal de voz!")
            return
        
        channel = ctx.author.voice.channel
        player = ctx.guild.voice_client
        
        if not player:
            player = await channel.connect(cls=wavelink.Player)
        
        tracks = await wavelink.Playable.search(query)
        if not tracks:
            await ctx.send(f"❌ No encontré: {query}")
            return
        
        track = tracks[0]
        
        if not player.playing:
            await player.play(track)
            embed = discord.Embed(
                title="🎵 Reproduciendo",
                description=f"**{track.title}**",
                color=0xFF69B4
            )
            await ctx.send(embed=embed)
        else:
            await player.queue.put_wait(track)
            embed = discord.Embed(
                title="➕ Agregada a cola",
                description=f"**{track.title}**\nPosición: {len(player.queue)}",
                color=0x00FF7F
            )
            await ctx.send(embed=embed)
            
    except Exception as e:
        logger.error(f"Error en play: {e}")
        await ctx.send("❌ Error reproduciendo música")

@commands.command(name='pause')
async def pause_command(ctx):
    """Pausar música"""
    player = ctx.guild.voice_client
    if player and player.playing:
        await player.pause(True)
        await ctx.send("⏸️ Música pausada")
    else:
        await ctx.send("❌ No hay música reproduciéndose")

@commands.command(name='resume')
async def resume_command(ctx):
    """Reanudar música"""
    player = ctx.guild.voice_client
    if player and player.paused:
        await player.pause(False)
        await ctx.send("▶️ Música reanudada")
    else:
        await ctx.send("❌ La música no está pausada")

@commands.command(name='skip', aliases=['next'])
async def skip_command(ctx):
    """Saltar canción"""
    player = ctx.guild.voice_client
    if player and player.playing:
        await player.skip()
        await ctx.send("⏭️ Canción saltada")
    else:
        await ctx.send("❌ No hay música para saltar")

@commands.command(name='stop')
async def stop_command(ctx):
    """Detener música"""
    player = ctx.guild.voice_client
    if player:
        await player.disconnect()
        await ctx.send("⏹️ Desconectada. ¡Sayonara! 💕")
    else:
        await ctx.send("❌ No estoy en un canal de voz")

@commands.command(name='volume', aliases=['vol'])
async def volume_command(ctx, volume: int = None):
    """Cambiar volumen"""
    player = ctx.guild.voice_client
    if not player:
        await ctx.send("❌ No hay reproductor activo")
        return
    
    if volume is None:
        await ctx.send(f"🔊 Volumen actual: {player.volume}%")
        return
    
    if not 1 <= volume <= 200:
        await ctx.send("❌ El volumen debe estar entre 1 y 200")
        return
    
    await player.set_volume(volume)
    await ctx.send(f"🔊 Volumen cambiado a {volume}%")

@commands.command(name='queue', aliases=['q'])
async def queue_command(ctx):
    """Mostrar cola"""
    player = ctx.guild.voice_client
    if not player:
        await ctx.send("❌ No hay reproductor activo")
        return
    
    if not player.queue:
        embed = discord.Embed(
            title="📝 Cola Vacía",
            description="No hay canciones en la cola",
            color=0xFFAA00
        )
        if player.current:
            embed.add_field(name="🎵 Reproduciendo", value=player.current.title, inline=False)
        await ctx.send(embed=embed)
        return
    
    queue_list = []
    for i, track in enumerate(list(player.queue)[:10]):
        queue_list.append(f"`{i+1}.` **{track.title}**")
    
    embed = discord.Embed(
        title="📝 Cola de Reproducción",
        description="\n".join(queue_list),
        color=0xFF69B4
    )
    
    if len(player.queue) > 10:
        embed.set_footer(text=f"Mostrando 10 de {len(player.queue)} canciones")
    
    await ctx.send(embed=embed)

@commands.command(name='np', aliases=['nowplaying'])
async def nowplaying_command(ctx):
    """Canción actual"""
    player = ctx.guild.voice_client
    if not player or not player.current:
        await ctx.send("❌ No hay música reproduciéndose")
        return
    
    track = player.current
    duration_ms = track.length
    minutes, seconds = divmod(duration_ms // 1000, 60)
    duration_str = f"{minutes}:{seconds:02d}"
    
    embed = discord.Embed(
        title="🎵 Reproduciendo Ahora",
        description=f"**{track.title}**",
        color=0xFF69B4
    )
    
    embed.add_field(name="⏱️ Duración", value=duration_str, inline=True)
    embed.add_field(name="🔊 Volumen", value=f"{player.volume}%", inline=True)
    embed.add_field(name="📝 En Cola", value=f"{len(player.queue)}", inline=True)
    embed.add_field(name="🎚️ Filtros", value="Calidad HD", inline=True)
    
    await ctx.send(embed=embed)

@commands.command(name='help', aliases=['comandos'])
async def help_command(ctx):
    """Ayuda"""
    embed = discord.Embed(
        title="🌸 Comandos de SakuraBot",
        description="Lista completa de comandos disponibles:",
        color=0xFF69B4
    )
    
    embed.add_field(
        name="🎵 Música",
        value="`?play <canción>` - Reproducir música\n"
              "`?pause` - Pausar música\n"
              "`?resume` - Reanudar música\n"
              "`?skip` - Saltar canción\n"
              "`?stop` - Detener y desconectar\n"
              "`?volume <1-200>` - Cambiar volumen\n"
              "`?queue` - Ver cola\n"
              "`?np` - Canción actual",
        inline=False
    )
    
    embed.add_field(
        name="🎚️ Características",
        value="• Codificador de audio HD\n"
              "• Filtros de calidad automáticos\n"
              "• Fallback inteligente de nodos\n"
              "• Optimización de graves y agudos",
        inline=False
    )
    
    await ctx.send(embed=embed)

async def main():
    """Función principal"""
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("No DISCORD_TOKEN found")
        return
    
    bot = SakuraBot()
    
    # Agregar comandos
    bot.add_command(play_command)
    bot.add_command(pause_command)
    bot.add_command(resume_command)
    bot.add_command(skip_command)
    bot.add_command(stop_command)
    bot.add_command(volume_command)
    bot.add_command(queue_command)
    bot.add_command(nowplaying_command)
    bot.add_command(help_command)
    
    try:
        await bot.start(token)
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main())