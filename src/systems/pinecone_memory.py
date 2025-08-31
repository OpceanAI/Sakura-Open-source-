"""
Pinecone Vector Database Integration for Discord Bot AI Memory
Provides persistent conversation memory and semantic search
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import openai
from pinecone import Pinecone, ServerlessSpec
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConversationMemory:
    user_id: str
    guild_id: str
    message: str
    response: str
    timestamp: datetime
    context: Dict[str, Any]
    
class PineconeMemoryManager:
    """
    Advanced AI Memory System using Pinecone Vector Database
    Enables contextual conversation memory and semantic search
    """
    
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.index_name = "sakura-bot-memory"
        self.pc = None
        self.index = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize Pinecone connection and create index if needed"""
        try:
            if not self.pinecone_api_key:
                logger.warning("⚠️ PINECONE_API_KEY not found - memory features disabled")
                return False
                
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                logger.info("🔧 Creating Pinecone index for bot memory...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                await asyncio.sleep(15)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            self.initialized = True
            
            logger.info("✅ Pinecone AI Memory System initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            return False
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            if not self.openai_api_key:
                return []
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = await asyncio.to_thread(
                lambda: client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text.replace("\n", " ")
                )
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def store_conversation(self, user_id: str, guild_id: str, 
                               message: str, bot_response: str, 
                               context: Dict[str, Any] = None):
        """Store conversation in vector database with semantic embeddings"""
        if not self.initialized:
            return False
            
        try:
            # Create combined text for embedding
            conversation_text = f"User: {message}\nBot: {bot_response}"
            
            # Generate embedding
            embedding = await self.get_embedding(conversation_text)
            if not embedding:
                return False
            
            # Create unique ID
            conversation_id = f"{user_id}_{guild_id}_{int(datetime.now().timestamp())}"
            
            # Prepare metadata
            metadata = {
                "user_id": user_id,
                "guild_id": guild_id,
                "message": message[:500],  # Truncate for metadata
                "response": bot_response[:500],
                "timestamp": datetime.now().isoformat(),
                "context": json.dumps(context or {})[:1000]
            }
            
            # Store in Pinecone
            await asyncio.to_thread(
                self.index.upsert,
                vectors=[(conversation_id, embedding, metadata)]
            )
            
            logger.info(f"💾 Stored conversation memory for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return False
    
    async def get_relevant_memories(self, user_id: str, guild_id: str, 
                                  current_message: str, 
                                  limit: int = 5) -> List[ConversationMemory]:
        """Retrieve relevant conversation memories using semantic search"""
        if not self.initialized:
            return []
            
        try:
            # Generate embedding for current message
            query_embedding = await self.get_embedding(current_message)
            if not query_embedding:
                return []
            
            # Search for similar conversations
            query_results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                filter={
                    "user_id": {"$eq": user_id},
                    "guild_id": {"$eq": guild_id}
                },
                top_k=limit,
                include_metadata=True
            )
            
            # Convert results to ConversationMemory objects
            memories = []
            for match in query_results['matches']:
                if match['score'] > 0.7:  # Only include relevant matches
                    metadata = match['metadata']
                    memory = ConversationMemory(
                        user_id=metadata['user_id'],
                        guild_id=metadata['guild_id'],
                        message=metadata['message'],
                        response=metadata['response'],
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        context=json.loads(metadata.get('context', '{}'))
                    )
                    memories.append(memory)
            
            logger.info(f"🧠 Retrieved {len(memories)} relevant memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def get_user_conversation_history(self, user_id: str, guild_id: str, 
                                          days: int = 7) -> List[ConversationMemory]:
        """Get recent conversation history for a user"""
        if not self.initialized:
            return []
            
        try:
            # Calculate timestamp filter
            since_date = datetime.now() - timedelta(days=days)
            
            # Query recent conversations
            query_results = await asyncio.to_thread(
                self.index.query,
                vector=[0.0] * 1536,  # Dummy vector for metadata-only search
                filter={
                    "user_id": {"$eq": user_id},
                    "guild_id": {"$eq": guild_id},
                    "timestamp": {"$gte": since_date.isoformat()}
                },
                top_k=20,
                include_metadata=True
            )
            
            memories = []
            for match in query_results['matches']:
                metadata = match['metadata']
                memory = ConversationMemory(
                    user_id=metadata['user_id'],
                    guild_id=metadata['guild_id'],
                    message=metadata['message'],
                    response=metadata['response'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    context=json.loads(metadata.get('context', '{}'))
                )
                memories.append(memory)
            
            # Sort by timestamp
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def search_conversations(self, query: str, user_id: str = None, 
                                 guild_id: str = None, limit: int = 10) -> List[ConversationMemory]:
        """Search conversations using natural language query"""
        if not self.initialized:
            return []
            
        try:
            # Generate embedding for search query
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Build filter
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
            if guild_id:
                filter_dict["guild_id"] = {"$eq": guild_id}
            
            # Search conversations
            query_results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=limit,
                include_metadata=True
            )
            
            memories = []
            for match in query_results['matches']:
                if match['score'] > 0.6:  # Similarity threshold
                    metadata = match['metadata']
                    memory = ConversationMemory(
                        user_id=metadata['user_id'],
                        guild_id=metadata['guild_id'],
                        message=metadata['message'],
                        response=metadata['response'],
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        context=json.loads(metadata.get('context', '{}'))
                    )
                    memories.append(memory)
            
            logger.info(f"🔍 Found {len(memories)} conversations matching: {query}")
            return memories
            
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        if not self.initialized:
            return {"error": "Pinecone not initialized"}
            
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {}),
                "status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def clear_user_memories(self, user_id: str, guild_id: str = None):
        """Clear all memories for a specific user"""
        if not self.initialized:
            return False
            
        try:
            # Build filter for deletion
            filter_dict = {"user_id": {"$eq": user_id}}
            if guild_id:
                filter_dict["guild_id"] = {"$eq": guild_id}
            
            # Delete vectors matching filter
            await asyncio.to_thread(
                self.index.delete,
                filter=filter_dict
            )
            
            logger.info(f"🗑️ Cleared memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False

# Enhanced AI response with memory context
async def get_ai_response_with_memory(memory_manager: PineconeMemoryManager,
                                    user_id: str, guild_id: str, 
                                    message: str, ai_provider) -> str:
    """Generate AI response enhanced with conversation memory"""
    try:
        # Get relevant memories
        memories = await memory_manager.get_relevant_memories(
            user_id, guild_id, message, limit=3
        )
        
        # Build context from memories
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant conversation history:\n"
            for memory in memories:
                memory_context += f"- User said: {memory.message}\n  You responded: {memory.response}\n"
        
        # Enhanced prompt with memory
        enhanced_prompt = f"{message}{memory_context}"
        
        # Get AI response
        response = await ai_provider.get_response(enhanced_prompt, user_id)
        
        # Store new conversation
        await memory_manager.store_conversation(
            user_id, guild_id, message, response,
            context={"has_memory": len(memories) > 0}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Memory-enhanced response failed: {e}")
        return await ai_provider.get_response(message, user_id)

# Global memory manager instance
memory_manager = PineconeMemoryManager()