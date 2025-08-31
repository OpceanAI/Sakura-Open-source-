# Sakura IA - Advanced Discord Bot

## Overview
Sakura IA is an advanced Discord bot designed for exceptional AI integration and intelligent ensemble systems. It provides a robust, highly available, and feature-rich Discord bot experience, encompassing music playback, moderation, and advanced AI interactions. The project features a comprehensive kawaii personality with pastel aesthetics and advanced multimodal capabilities.

## User Preferences
### Development Style
- **Focus**: Functionality over aesthetics, stability over novelty.
- **Architecture**: Robust systems with multiple backups.
- **AI**: Preference for high-quality free models (DeepSeek R1, Gemini).
- **Logging**: Detailed for debugging and monitoring.

### Bot Personality
- **Sakura IA**: Kawaii personality with pastel emojis (🌸✨💖).
- **Language**: Spanish as the primary language.
- **Tone**: Friendly, affectionate, use of "UwU", ">w<", terms like "mi amor", "senpai".

## System Architecture

Sakura IA is a comprehensive Discord bot with advanced capabilities:

**Main Bot**: Python-based Discord bot (`bot_unificado_completo.py`) featuring:
- Advanced AI ensemble system with multiple AI providers
- Comprehensive music system with Lavalink integration
- Multimodal content processing (text, image, audio)
- Kawaii personality with pastel emojis and Spanish interactions

### Core System Features:
*   **Intelligent AI Ensemble System**: Dynamically alternates between various AI providers (Gemini with Internet Search, DeepSeek R1) with multiple methods (free API, local Transformers, Hugging Face API). Features enhanced Gemini integration with automatic web search capabilities for real-time context. It includes a robust four-level fallback system (Mistral Free + kawaii responses) to ensure continuous operation, augmented with a Cloudflare AI (Level 0) for edge processing, forming a 5-level ensemble.
*   **Music System**: Configured with multiple Lavalink nodes for optimal performance across different regions, supporting YouTube, Spotify, SoundCloud and other sources with advanced audio quality filters and equalizers
*   **Database Management**: Employs PostgreSQL as the primary database, Redis for intelligent caching and session management, and Pinecone for AI vector memory. SQLite is used for specific local functions.
*   **Moderation and Management**: Features an intelligent automod system, advanced role management, comprehensive server activity tracking, and an integrated ticket system with welcome/farewell messages.
*   **UI/UX Decisions**: The bot's "kawaii" personality is consistently applied across commands, error messages, and interactive elements, using pastel emojis (🌙✨💫🌸🥺) and a timid conversational style.
*   **Deployment Configuration**: Single bot deployment with console-only operation for simplified management.
*   **Image Analysis**: Implements a comprehensive multimodal image analysis system with commands for general analysis, description, and OCR, leveraging Gemini Vision and various fallbacks.
*   **Sistema de Ensamblaje Multimodal**: Advanced multimodal detection and assembly system that can automatically detect and analyze audio, text, and images together. Features intelligent content type detection, audio transcription with speech recognition, advanced image analysis with multiple FREE AI providers (Gemini, DeepSeek, Cloudflare AI - NO OpenAI/Anthropic), and comprehensive multimodal content assembly with AI-powered analysis combining all input types. Accessible through `/ensamblar_contenido` command. **Updated January 2025: Completely removed OpenAI and Anthropic dependencies**.
*   **Enhanced AI Context**: Gemini AI now features automatic internet search integration, intelligently detecting when queries would benefit from current information and incorporating real-time web search results for more accurate and up-to-date responses.
*   **Command Structure**: Operates exclusively with slash commands (`/`).
*   **Response Length Management**: Implements intelligent response truncation system to handle Discord's 2000 character limit, with emergency fallbacks and kawaii truncation messages.
*   **Cooldown Systems**: Implements user-specific cooldowns for AI interactions (default 30 seconds) and social interactions (default 10 seconds), with administrative controls.

## External Dependencies

*   **Discord API**: For core bot functionalities and interactions.
*   **Lavalink**: Private server for music playback.
*   **PostgreSQL**: Primary database.
*   **Redis**: Caching and session management.
*   **Pinecone**: Vector database for AI memory.
*   **SQLite**: Local storage.
*   **OpenRouter API**: For DeepSeek R1 (free tier) and other vision models.
*   **Google Gemini API**: Using latest `@google/genai` SDK v1.14.0 for Gemini 2.5 Flash/Pro with built-in thinking capabilities and multimodal support.
*   **Hugging Face API**: For additional transformer models and inference.
*   **Cloudflare**: Integrated for various services including Workers AI (for edge processing), Web Analytics, DNS, SSL/TLS, WAF, CDN, Pages, R2, and Stream.
*   **Anthropic, OpenAI, XAI**: Configured as backup AI providers.