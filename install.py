#!/usr/bin/env python3
"""
🌸✨ Sakura IA - Installation Script ✨🌸
====================================
Script automático de instalación y configuración
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 11):
        print("❌ Error: Se requiere Python 3.11 o superior")
        print(f"   Versión actual: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")

def check_dependencies():
    """Verificar e instalar dependencias"""
    print("🔧 Verificando dependencias...")
    
    try:
        import discord
        print("✅ discord.py está instalado")
    except ImportError:
        print("📦 Instalando discord.py...")
        subprocess.run([sys.executable, "-m", "pip", "install", "discord.py"], check=True)
    
    try:
        import psycopg2
        print("✅ psycopg2 está instalado")
    except ImportError:
        print("📦 Instalando psycopg2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psycopg2-binary"], check=True)

def setup_environment():
    """Configurar archivo de entorno"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔧 Creando archivo .env desde .env.example...")
        shutil.copy(env_example, env_file)
        print("✅ Archivo .env creado")
        print("⚠️  IMPORTANTE: Edita .env con tus credenciales antes de ejecutar el bot")
    elif env_file.exists():
        print("✅ Archivo .env ya existe")
    else:
        print("❌ Error: No se encontró .env.example")

def create_directories():
    """Crear directorios necesarios"""
    dirs = ["logs", "src/database"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio {dir_path} creado/verificado")

def main():
    """Función principal de instalación"""
    print("🌸✨ Iniciando instalación de Sakura IA ✨🌸")
    print("=" * 50)
    
    check_python_version()
    check_dependencies()
    setup_environment()
    create_directories()
    
    print("=" * 50)
    print("🎉 ¡Instalación completada!")
    print()
    print("📝 Próximos pasos:")
    print("1. Edita el archivo .env con tus credenciales")
    print("2. Configura tu base de datos PostgreSQL")
    print("3. Ejecuta: python bot_unificado_completo.py")
    print()
    print("🌸 ¡Que disfrutes a Sakura IA! ✨")

if __name__ == "__main__":
    main()