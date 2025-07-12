"""
Preprocessing module for NLP pipeline.

This module provides text cleaning, tokenization, and feature extraction utilities
for natural language processing tasks.
"""

from .text_cleaner import TextCleaner, MultilingualTextCleaner
from .tokenizer import (
    BasicTokenizer,
    BERTTokenizer,
    SubwordTokenizer,
    TokenizationOutput
)
from .feature_extractor import (
    TfidfFeatureExtractor,
    BERTFeatureExtractor,
    CountVectorFeatureExtractor,
    TopicFeatureExtractor,
    EmbeddingFeatureExtractor,
    FeatureOutput
)

__all__ = [
    # Text cleaning
    'TextCleaner',
    'MultilingualTextCleaner',
    
    # Tokenization
    'BasicTokenizer',
    'BERTTokenizer',
    'SubwordTokenizer',
    'TokenizationOutput',
    
    # Feature extraction
    'TfidfFeatureExtractor',
    'BERTFeatureExtractor',
    'CountVectorFeatureExtractor',
    'TopicFeatureExtractor',
    'EmbeddingFeatureExtractor',
    'FeatureOutput'
]

# Version info
__version__ = '0.1.0'