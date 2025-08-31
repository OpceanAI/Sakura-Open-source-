# Sakura IA - Advanced Discord Bot

## Overview

Sakura IA is a comprehensive, production-ready Discord bot featuring an advanced AI ensemble system with kawaii personality. The bot provides intelligent conversation capabilities through multiple AI providers (Gemini, DeepSeek R1, Cloudflare AI), a complete music system with Lavalink integration, sophisticated moderation tools, and multimodal content processing. The system is designed with high availability in mind, featuring multiple fallback mechanisms and robust error handling throughout.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Bot Architecture
**Main Bot Implementation**: Python-based Discord bot (`bot_unificado_completo.py`) with modular system design featuring comprehensive AI integration, music capabilities, and kawaii personality framework.

**AI Ensemble System**: 
- Multi-provider AI system with intelligent routing between Gemini (with web search), DeepSeek R1, and Cloudflare AI
- 5-level fallback system ensuring 99.9% AI response availability
- Vector memory system using Pinecone for persistent conversation context
- Automatic web search integration for Gemini responses requiring current information

**Music System Architecture**:
- Multiple Lavalink node configuration with automatic failover
- Support for YouTube, Spotify, SoundCloud platforms
- Advanced audio codec system with quality filters and equalizers
- Regional optimization with 8+ public Lavalink nodes

**Database Design**:
- Primary: PostgreSQL for core data persistence
- Cache Layer: Redis for session management and rate limiting
- Vector Storage: Pinecone for AI memory and semantic search
- Local Storage: SQLite for specific functionality

**Moderation Framework**:
- Dual AutoMod system (Discord native + custom implementation)
- Intelligent content filtering with real-time processing
- Comprehensive role management and permission system
- Integrated ticket system with automated workflows

**Multimodal Processing**:
- Advanced content detection system for text, image, and audio
- OCR capabilities with multiple fallback engines
- Audio transcription and analysis
- Image analysis using Gemini Vision with multiple providers

### Technical Implementation Details
**Command Structure**: Exclusively slash commands (`/`) with comprehensive error handling and user feedback

**Personality System**: Consistent kawaii personality across all interactions using pastel emojis (🌸✨💖🥺💫) with Spanish language preference and timid conversational style

**Rate Limiting & Resilience**: Circuit breaker pattern implementation with intelligent rate limit management and bucket tracking to prevent API exhaustion

**Response Management**: Intelligent text truncation system handling Discord's 2000 character limit with graceful degradation and kawaii error messages

**Cooldown Systems**: User-specific cooldowns (30s AI, 10s social) with administrative override capabilities

## External Dependencies

**Core Services**:
- Discord API: Primary bot platform integration
- PostgreSQL: Main database for persistent data storage
- Redis: Caching, session management, and rate limiting
- Pinecone: Vector database for AI conversation memory

**AI Providers**:
- Google Gemini: Primary AI with web search integration
- DeepSeek R1: Secondary AI via OpenRouter (free tier)
- Cloudflare AI: Edge-optimized AI responses
- OpenAI API: Image generation and vision processing

**Music Infrastructure**:
- Lavalink: Private and public servers for music playback
- YouTube/Spotify/SoundCloud APIs: Music source integration
- Wavelink: Python library for Lavalink communication

**Additional Services**:
- Kaggle API: Dataset and model integration for enhanced AI
- Hugging Face: Transformer models and additional AI capabilities
- Google Search API: Real-time information retrieval
- Various image processing libraries: PIL, ImageEnhance, ImageFilter