"""
🌸 SakuraBot IA - Simple Working Music Bot 🌸
============================================
Una waifu kawaii que funciona con comandos slash y música
"""

import os
import discord
import wavelink
import asyncio
import logging
from discord.ext import commands
from discord import app_commands
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SakuraBot')

# 8 nodos Lavalink públicos ordenados por estabilidad
NODES = [
    {"host": "lava.luke.gg", "port": 443, "password": "discordbotlist.com", "secure": True},
    {"host": "46.202.82.164", "port": 1027, "password": "jmlitelavalink", "secure": False},
    {"host": "lavalink-v2.pericsq.ro", "port": 6677, "password": "wwweasycodero", "secure": False},
    {"host": "69.30.219.180", "port": 1047, "password": "yothisnodeishostedbymushroom0162", "secure": False},
    {"host": "lava3.horizxon.studio", "port": 80, "password": "horizxon.studio", "secure": False},
    {"host": "lava2.horizxon.studio", "port": 80, "password": "horizxon.studio", "secure": False},
    {"host": "lavalink.micium-hosting.com", "port": 80, "password": "micium-hosting.com", "secure": False},
    {"host": "lavalink.oops.wtf", "port": 443, "password": "www.freelavalink.ga", "secure": True}
]

class SakuraBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
        self.current_node = 0

    async def setup_hook(self):
        """Setup inicial del bot"""
        logger.info("🌸 Iniciando SakuraBot...")
        
        # Intentar conectar a nodos
        await self.connect_nodes()
        
        # Sincronizar comandos
        try:
            synced = await self.tree.sync()
            logger.info(f"🌸 Sincronizados {len(synced)} comandos")
        except Exception as e:
            logger.error(f"Error sincronizando: {e}")

    async def connect_nodes(self):
        """Conectar a nodos Lavalink"""
        for i, node_config in enumerate(NODES):
            try:
                protocol = "wss" if node_config["secure"] else "ws"
                uri = f"{protocol}://{node_config['host']}:{node_config['port']}"
                
                node = wavelink.Node(
                    uri=uri,
                    password=node_config["password"],
                    identifier=f"Node{i+1}"
                )
                
                await wavelink.Pool.connect(client=self, nodes=[node])
                logger.info(f"✅ Conectado a {node_config['host']}")
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

    async def on_wavelink_node_ready(self, payload):
        """Nodo listo"""
        logger.info(f"🎵 UwU conectada a {payload.node.identifier}! Calidad asegurada~")

# Comandos slash
@app_commands.describe(query="Canción a reproducir")
async def play(interaction: discord.Interaction, query: str):
    """Reproducir música"""
    try:
        await interaction.response.defer()
        
        if not interaction.user.voice:
            embed = discord.Embed(
                title="❌ Error",
                description="¡Necesitas estar en un canal de voz! UwU",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        # Conectar al canal
        channel = interaction.user.voice.channel
        player = interaction.guild.voice_client
        
        if not player:
            player = await channel.connect(cls=wavelink.Player)
        
        # Buscar música
        tracks = await wavelink.Playable.search(query)
        if not tracks:
            embed = discord.Embed(
                title="❌ No encontrado",
                description=f"No encontré: {query}",
                color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return
        
        track = tracks[0]
        
        # Reproducir
        if not player.playing:
            await player.play(track)
            # Aplicar filtros de calidad
            try:
                eq_bands = [
                    {"band": 0, "gain": -0.20},  # Graves
                    {"band": 1, "gain": 0.10},   # Bajo-medio
                    {"band": 4, "gain": 0.05},   # Medio  
                    {"band": 7, "gain": -0.30},  # Agudos
                ]
                filters = wavelink.Filters(equalizer=eq_bands)
                await player.set_filters(filters)
                await player.set_volume(75)
            except:
                pass
            
            embed = discord.Embed(
                title="🎵 Reproduciendo",
                description=f"**{track.title}**",
                color=0xFF69B4
            )
        else:
            await player.queue.put_wait(track)
            embed = discord.Embed(
                title="➕ Agregada a cola",
                description=f"**{track.title}**",
                color=0x00FF7F
            )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error en play: {e}")
        embed = discord.Embed(
            title="❌ Error",
            description="Hubo un problema con la música",
            color=0xFF0000
        )
        await interaction.followup.send(embed=embed)

@app_commands.describe()
async def pause(interaction: discord.Interaction):
    """Pausar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.playing:
        embed = discord.Embed(
            title="❌ Error",
            description="No hay música reproduciéndose",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(True)
    embed = discord.Embed(
        title="⏸️ Pausada",
        description="Música pausada UwU",
        color=0xFFAA00
    )
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def resume(interaction: discord.Interaction):
    """Reanudar música"""
    player = interaction.guild.voice_client
    
    if not player or not player.paused:
        embed = discord.Embed(
            title="❌ Error", 
            description="La música no está pausada",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.pause(False)
    embed = discord.Embed(
        title="▶️ Reanudada",
        description="Música reanudada! Kyaa~",
        color=0x00FF7F
    )
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def skip(interaction: discord.Interaction):
    """Saltar canción"""
    player = interaction.guild.voice_client
    
    if not player or not player.playing:
        embed = discord.Embed(
            title="❌ Error",
            description="No hay música para saltar",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.skip()
    embed = discord.Embed(
        title="⏭️ Saltada",
        description="Canción saltada! >w<",
        color=0x00FF7F
    )
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def stop(interaction: discord.Interaction):
    """Detener música"""
    player = interaction.guild.voice_client
    
    if not player:
        embed = discord.Embed(
            title="❌ Error",
            description="No estoy en un canal de voz",
            color=0xFF0000
        )
        await interaction.response.send_message(embed=embed)
        return
    
    await player.disconnect()
    embed = discord.Embed(
        title="⏹️ Detenida",
        description="Sayonara! Gracias por la música 💕",
        color=0xFF69B4
    )
    await interaction.response.send_message(embed=embed)

@app_commands.describe()
async def help_command(interaction: discord.Interaction):
    """Mostrar ayuda"""
    embed = discord.Embed(
        title="🌸 Comandos de SakuraBot",
        description="¡Ohayo! Aquí están mis comandos UwU",
        color=0xFF69B4
    )
    
    embed.add_field(
        name="🎵 Música",
        value="`/play` - Reproducir música\n"
              "`/pause` - Pausar\n"
              "`/resume` - Reanudar\n"
              "`/skip` - Saltar\n"
              "`/stop` - Detener",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# Configurar bot
async def main():
    bot = SakuraBot()
    
    # Agregar comandos
    bot.tree.add_command(app_commands.Command(name="play", description="🎵 Reproducir música", callback=play))
    bot.tree.add_command(app_commands.Command(name="pause", description="⏸️ Pausar música", callback=pause))
    bot.tree.add_command(app_commands.Command(name="resume", description="▶️ Reanudar música", callback=resume))
    bot.tree.add_command(app_commands.Command(name="skip", description="⏭️ Saltar canción", callback=skip))
    bot.tree.add_command(app_commands.Command(name="stop", description="⏹️ Detener música", callback=stop))
    bot.tree.add_command(app_commands.Command(name="help", description="❓ Mostrar ayuda", callback=help_command))
    
    # Iniciar bot
    token = os.getenv('DISCORD_TOKEN')
    if token:
        await bot.start(token)
    else:
        logger.error("No DISCORD_TOKEN found")

if __name__ == "__main__":
    asyncio.run(main())