#!/usr/bin/env python3
"""
Kaggle Integration for Sakura Discord Bot
Provides access to datasets, models, and enhanced AI capabilities
"""

import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import zipfile
import requests
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleAIIntegration:
    """
    Advanced Kaggle integration for Discord bot AI enhancement
    """
    
    def __init__(self):
        self.username = os.getenv('KAGGLE_USERNAME')
        self.api_key = os.getenv('KAGGLE_KEY')
        self.base_url = "https://www.kaggle.com/api/v1"
        self.data_dir = Path("kaggle_data")
        self.models_dir = Path("kaggle_models")
        self.cache_duration = timedelta(hours=24)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize session
        self.session = requests.Session()
        if self.username and self.api_key:
            self.session.auth = (self.username, self.api_key)
        
        logger.info(f"🔧 Kaggle integration initialized for user: {self.username}")
    
    async def verify_connection(self) -> bool:
        """Verify Kaggle API connection"""
        try:
            response = self.session.get(f"{self.base_url}/datasets/list")
            if response.status_code == 200:
                logger.info("✅ Kaggle API connection verified")
                return True
            else:
                logger.error(f"❌ Kaggle API error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Kaggle connection failed: {e}")
            return False
    
    async def get_popular_datasets(self, search_term: str = "", page: int = 1, max_size: int = 1000) -> List[Dict]:
        """Get popular datasets from Kaggle"""
        try:
            params = {
                'group': 'public',
                'sort': 'hottest',
                'search': search_term,
                'page': page,
                'maxSize': max_size
            }
            
            response = self.session.get(f"{self.base_url}/datasets/list", params=params)
            if response.status_code == 200:
                datasets = response.json()
                logger.info(f"📊 Found {len(datasets)} datasets for '{search_term}'")
                return datasets
            else:
                logger.error(f"❌ Failed to fetch datasets: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"❌ Error fetching datasets: {e}")
            return []
    
    async def download_dataset(self, dataset_ref: str, force_download: bool = False) -> Optional[Path]:
        """Download a dataset from Kaggle"""
        try:
            dataset_path = self.data_dir / dataset_ref.replace('/', '_')
            
            # Check if already downloaded and not expired
            if dataset_path.exists() and not force_download:
                mod_time = datetime.fromtimestamp(dataset_path.stat().st_mtime)
                if datetime.now() - mod_time < self.cache_duration:
                    logger.info(f"📁 Using cached dataset: {dataset_ref}")
                    return dataset_path
            
            logger.info(f"⬇️ Downloading dataset: {dataset_ref}")
            
            # Download dataset
            response = self.session.get(f"{self.base_url}/datasets/download/{dataset_ref}")
            if response.status_code == 200:
                zip_path = dataset_path.with_suffix('.zip')
                
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract if it's a zip file
                if zipfile.is_zipfile(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    zip_path.unlink()  # Remove zip file
                
                logger.info(f"✅ Dataset downloaded: {dataset_path}")
                return dataset_path
            else:
                logger.error(f"❌ Failed to download dataset: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            return None
    
    async def get_sentiment_model_data(self) -> Optional[pd.DataFrame]:
        """Get sentiment analysis training data"""
        try:
            # Try to get Twitter sentiment dataset
            datasets = await self.get_popular_datasets("twitter sentiment", max_size=100)
            if not datasets:
                datasets = await self.get_popular_datasets("sentiment analysis", max_size=100)
            
            if datasets:
                # Use the first available dataset
                dataset_ref = datasets[0]['ref']
                dataset_path = await self.download_dataset(dataset_ref)
                
                if dataset_path:
                    # Try to find CSV files
                    csv_files = list(dataset_path.glob("*.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        logger.info(f"📊 Loaded sentiment data: {len(df)} rows")
                        return df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiment data: {e}")
            return None
    
    async def get_discord_chat_data(self) -> Optional[pd.DataFrame]:
        """Get Discord chat/conversation data for training"""
        try:
            datasets = await self.get_popular_datasets("discord chat", max_size=500)
            if not datasets:
                datasets = await self.get_popular_datasets("chat conversation", max_size=500)
            
            if datasets:
                dataset_ref = datasets[0]['ref']
                dataset_path = await self.download_dataset(dataset_ref)
                
                if dataset_path:
                    csv_files = list(dataset_path.glob("*.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        logger.info(f"💬 Loaded chat data: {len(df)} rows")
                        return df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading chat data: {e}")
            return None
    
    async def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using Kaggle data"""
        try:
            # Simple sentiment analysis based on keywords
            # In a real implementation, you'd use a trained model
            positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'excellent', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disgusting', 'worst']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
            
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = max(0, 1 - positive_score - negative_score)
            
            return {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
    
    async def get_moderation_keywords(self) -> List[str]:
        """Get moderation keywords from Kaggle datasets"""
        try:
            datasets = await self.get_popular_datasets("toxic comments", max_size=100)
            if not datasets:
                datasets = await self.get_popular_datasets("hate speech", max_size=100)
            
            if datasets:
                dataset_ref = datasets[0]['ref']
                dataset_path = await self.download_dataset(dataset_ref)
                
                if dataset_path:
                    csv_files = list(dataset_path.glob("*.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        
                        # Extract common toxic words (simplified)
                        toxic_words = set()
                        if 'comment_text' in df.columns:
                            toxic_comments = df[df.get('toxic', 0) == 1]['comment_text'].dropna()
                            for comment in toxic_comments.head(100):  # Limit for performance
                                words = comment.lower().split()
                                toxic_words.update([word for word in words if len(word) > 3])
                        
                        logger.info(f"🛡️ Loaded {len(toxic_words)} moderation keywords")
                        return list(toxic_words)[:500]  # Limit list size
            
            # Fallback basic keywords
            return ['spam', 'toxic', 'hate', 'abuse', 'harassment']
            
        except Exception as e:
            logger.error(f"❌ Error loading moderation keywords: {e}")
            return ['spam', 'toxic', 'hate', 'abuse', 'harassment']
    
    async def enhance_ai_response(self, user_message: str, bot_response: str) -> str:
        """Enhance bot response using Kaggle insights"""
        try:
            # Analyze sentiment of user message
            sentiment = await self.analyze_text_sentiment(user_message)
            
            # Adjust response based on sentiment
            if sentiment['negative'] > 0.7:
                # User seems upset, make response more empathetic
                empathy_prefixes = [
                    "Entiendo que puedas sentirte así, ",
                    "Lamento que te sientas de esa manera, ",
                    "Comprendo tu frustración, "
                ]
                import random
                prefix = random.choice(empathy_prefixes)
                bot_response = prefix + bot_response.lower()
                
            elif sentiment['positive'] > 0.7:
                # User seems happy, make response more enthusiastic
                enthusiasm_suffixes = [
                    " ✨ ¡Me alegra verte tan positivo!",
                    " 🌸 ¡Qué buena energía tienes!",
                    " 💖 ¡Me encanta tu actitud!"
                ]
                import random
                suffix = random.choice(enthusiasm_suffixes)
                bot_response = bot_response + suffix
            
            return bot_response
            
        except Exception as e:
            logger.error(f"❌ Error enhancing response: {e}")
            return bot_response
    
    async def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about available datasets"""
        try:
            stats = {
                'total_datasets': 0,
                'categories': {},
                'recent_datasets': [],
                'popular_datasets': []
            }
            
            # Get popular datasets
            datasets = await self.get_popular_datasets("", max_size=50)
            stats['total_datasets'] = len(datasets)
            
            for dataset in datasets[:10]:
                stats['popular_datasets'].append({
                    'title': dataset.get('title', 'Unknown'),
                    'ref': dataset.get('ref', ''),
                    'downloadCount': dataset.get('downloadCount', 0)
                })
            
            logger.info(f"📊 Dataset statistics compiled: {stats['total_datasets']} datasets")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting dataset stats: {e}")
            return {'error': str(e)}


class KaggleModelWrapper:
    """
    Wrapper for Kaggle models integration with Discord bot
    """
    
    def __init__(self, kaggle_integration: KaggleAIIntegration):
        self.kaggle = kaggle_integration
        self.loaded_models = {}
    
    async def classify_message_toxicity(self, message: str) -> float:
        """Classify message toxicity using Kaggle data"""
        try:
            # Get moderation keywords
            if 'moderation_keywords' not in self.loaded_models:
                keywords = await self.kaggle.get_moderation_keywords()
                self.loaded_models['moderation_keywords'] = set(keywords)
            
            keywords = self.loaded_models['moderation_keywords']
            
            # Simple classification based on keyword matching
            message_lower = message.lower()
            toxic_count = sum(1 for keyword in keywords if keyword in message_lower)
            
            # Normalize score
            total_words = len(message.split())
            if total_words == 0:
                return 0.0
            
            toxicity_score = min(1.0, toxic_count / total_words * 2)  # Scale up
            
            return toxicity_score
            
        except Exception as e:
            logger.error(f"❌ Error classifying toxicity: {e}")
            return 0.0
    
    async def suggest_response_style(self, user_message: str) -> str:
        """Suggest response style based on user message analysis"""
        try:
            sentiment = await self.kaggle.analyze_text_sentiment(user_message)
            
            if sentiment['negative'] > 0.6:
                return "empathetic"
            elif sentiment['positive'] > 0.6:
                return "enthusiastic"
            elif "question" in user_message.lower() or "?" in user_message:
                return "informative"
            else:
                return "casual"
                
        except Exception as e:
            logger.error(f"❌ Error suggesting response style: {e}")
            return "casual"


# Initialize global instance
kaggle_ai = None

async def initialize_kaggle_integration():
    """Initialize Kaggle integration"""
    global kaggle_ai
    try:
        kaggle_ai = KaggleAIIntegration()
        connection_ok = await kaggle_ai.verify_connection()
        
        if connection_ok:
            logger.info("🚀 Kaggle AI integration ready!")
            return kaggle_ai
        else:
            logger.error("❌ Failed to initialize Kaggle integration")
            return None
    except Exception as e:
        logger.error(f"❌ Error initializing Kaggle: {e}")
        return None

async def get_kaggle_enhanced_response(user_message: str, base_response: str) -> str:
    """Get Kaggle enhanced response"""
    global kaggle_ai
    
    if kaggle_ai is None:
        kaggle_ai = await initialize_kaggle_integration()
        if kaggle_ai is None:
            return base_response
    
    try:
        enhanced_response = await kaggle_ai.enhance_ai_response(user_message, base_response)
        return enhanced_response
    except Exception as e:
        logger.error(f"❌ Error getting enhanced response: {e}")
        return base_response

if __name__ == "__main__":
    # Test the integration
    async def test_kaggle():
        integration = await initialize_kaggle_integration()
        if integration:
            stats = await integration.get_dataset_stats()
            print(f"📊 Kaggle Stats: {stats}")
            
            sentiment = await integration.analyze_text_sentiment("I love this bot!")
            print(f"😊 Sentiment: {sentiment}")
    
    asyncio.run(test_kaggle())