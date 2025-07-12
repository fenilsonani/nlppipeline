"""NLP Models Module.

This module provides BERT-based sentiment analysis and spaCy-based entity extraction
with model loading, caching, and batch processing capabilities.
"""

from .base_model import BaseModel
from .sentiment_analyzer import SentimentAnalyzer
from .entity_extractor import EntityExtractor
from .model_manager import ModelManager

__all__ = [
    "BaseModel",
    "SentimentAnalyzer", 
    "EntityExtractor",
    "ModelManager"
]

# Module version
__version__ = "1.0.0"