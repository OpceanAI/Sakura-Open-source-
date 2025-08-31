# 🌸✨ Sakura IA - Advanced Discord Bot ✨🌸

<div align="center">
  <h3>Una asistente súper kawaii y adorable con personalidad adaptativa ♡</h3>
  <p>Sistema de afecto, búsquedas mágicas y diversión sin límites ♡</p>
  <p>Con los colores más bonitos rosa pastel del mundo ♡</p>
</div>

---

## 🌟 Características Principales

### 🤖 Sistema de IA Avanzado
- **Ensemble de IA Inteligente**: Múltiples proveedores (Gemini, DeepSeek R1, Cloudflare AI)
- **Sistema de Respaldo**: 5 niveles de fallback para garantizar disponibilidad
- **Búsqueda Web Automática**: Gemini con integración de búsquedas en tiempo real
- **Memoria Vectorial**: Sistema de memoria persistente con Pinecone

### 🎵 Sistema de Música Completo
- **Múltiples Nodos Lavalink**: Optimizado para diferentes regiones
- **Soporte Multi-Plataforma**: YouTube, Spotify, SoundCloud
- **Filtros de Audio**: Ecualizadores y efectos avanzados
- **Cola Inteligente**: Gestión avanzada de reproducción

### 🌸 Personalidad Kawaii
- **Emojis Pastel**: 🌸✨💖🥺💫
- **Idioma**: Español como idioma principal
- **Expresiones**: "UwU", ">w<", "mi amor", "senpai"
- **Tono**: Amigable, afectuoso y tímido

### 🛡️ Moderación Avanzada
- **AutoMod Inteligente**: Sistema de filtrado automático
- **Gestión de Roles**: Administración completa de permisos
- **Sistema de Tickets**: Soporte integrado
- **Seguimiento de Actividad**: Monitoreo completo del servidor

### 🎮 Sistemas Adicionales
- **Minijuegos**: Juegos interactivos integrados
- **Sistema de Logros**: Progresión y recompensas
- **Arte IA**: Comandos de generación artística
- **Análisis Multimodal**: Procesamiento de texto, imagen y audio

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.11+
- PostgreSQL
- Redis (opcional pero recomendado)
- Token de Discord Bot

### 1️⃣ Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/sakura-ia.git
cd sakura-ia
```

### 2️⃣ Configurar Variables de Entorno
```bash
cp .env.example .env
# Edita .env con tus credenciales
```

### 3️⃣ Instalar Dependencias
```bash
pip install -r pyproject.toml
```

### 4️⃣ Ejecutar el Bot
```bash
python bot_unificado_completo.py
```

## ⚙️ Configuración Avanzada

### Variables de Entorno Críticas
```env
# Discord (OBLIGATORIO)
DISCORD_TOKEN=tu_token_de_discord

# IA (Al menos uno requerido)
GEMINI_API_KEY=tu_api_key_de_gemini
OPENROUTER_API_KEY=tu_api_key_de_openrouter

# Base de Datos (OBLIGATORIO)
DATABASE_URL=postgresql://usuario:password@host:puerto/database
```

### Configuración de Base de Datos
El bot utiliza PostgreSQL como base de datos principal. Las tablas se crean automáticamente en el primer arranque.

### Configuración de Redis
Redis se utiliza para caché y gestión de sesiones. Es opcional pero altamente recomendado para mejor rendimiento.

## 📁 Estructura del Proyecto

```
sakura-ia/
├── bot_unificado_completo.py    # Bot principal
├── src/
│   ├── systems/                 # Sistemas de soporte
│   │   ├── redis_manager.py     # Gestión de Redis
│   │   ├── pinecone_memory.py   # Memoria vectorial
│   │   ├── kaggle_integration.py # Integración Kaggle
│   │   ├── cloudflare_ai.py     # IA de Cloudflare
│   │   └── multimodal_assembly_system.py # Sistema multimodal
│   ├── core/                    # Sistemas centrales
│   │   ├── achievements_system.py # Sistema de logros
│   │   ├── minigames_system.py  # Minijuegos
│   │   └── art_ai_commands.py   # Comandos de arte
│   ├── bots/                    # Versiones alternativas
│   ├── database/                # Bases de datos SQLite
│   └── docs/                    # Documentación
├── logs/                        # Archivos de log
├── .env.example                 # Plantilla de variables
└── README.md                    # Este archivo
```

## 🎯 Comandos Principales

### 🤖 IA y Conversación
- `/ask [pregunta]` - Hacer una pregunta a Sakura IA
- `/buscar [término]` - Búsqueda web con IA
- `/ensamblar_contenido` - Análisis multimodal de contenido

### 🎵 Música
- `/play [canción]` - Reproducir música
- `/queue` - Ver cola de reproducción
- `/skip` - Saltar canción
- `/stop` - Detener música

### 🎮 Diversión
- `/abrazo [@usuario]` - Abrazar a alguien
- `/beso [@usuario]` - Besar a alguien
- `/jugar` - Acceder a minijuegos

### 🛡️ Moderación
- `/ban [@usuario]` - Banear usuario
- `/kick [@usuario]` - Expulsar usuario
- `/clear [cantidad]` - Limpiar mensajes

## 🔧 Desarrollo

### Requisitos de Desarrollo
- Todas las dependencias están en `pyproject.toml`
- Configuración de logging avanzada
- Sistema de circuit breaker para rate limits
- Manejo inteligente de errores

### Contribuir
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📊 Estado del Proyecto

- ✅ Sistema de IA Funcional
- ✅ Música Completamente Operativa
- ✅ Moderación Avanzada
- ✅ Personalidad Kawaii Completa
- ✅ Análisis Multimodal
- ✅ Sistema de Memoria Vectorial

## 🆘 Soporte

Si tienes problemas o preguntas:
1. Revisa la documentación en `docs/`
2. Verifica tu configuración en `.env`
3. Consulta los logs en `logs/`
4. Abre un issue en GitHub

## 📄 Licencia

Este proyecto está bajo una licencia personalizada. Ver archivo LICENSE para más detalles.

---

<div align="center">
  <sub>Hecho con 💖 por el equipo de Sakura IA</sub>
</div>
