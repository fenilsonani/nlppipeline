"""
Text cleaning and normalization utilities for NLP preprocessing.
"""

import re
import string
from typing import List, Optional, Set, Union
import unicodedata
from functools import lru_cache


class TextCleaner:
    """Handles text cleaning and normalization operations."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize TextCleaner with configurable options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            remove_numbers: Remove numeric characters
            remove_punctuation: Remove punctuation marks
            remove_extra_whitespace: Normalize whitespace
            normalize_unicode: Normalize unicode characters
            custom_stopwords: Set of custom stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.custom_stopwords = custom_stopwords or set()
        
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean(self, text: str) -> str:
        """
        Clean a single text string according to configured options.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
            
        # Normalize unicode first if enabled
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
            
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
            
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Remove special characters
        if self.remove_special_chars and not self.remove_punctuation:
            text = self.special_char_pattern.sub(' ', text)
            
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
            
        # Remove custom stopwords
        if self.custom_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.custom_stopwords]
            text = ' '.join(words)
            
        return text
    
    def clean_batch(self, texts: List[str], batch_size: int = 1000) -> List[str]:
        """
        Clean multiple texts efficiently in batches.
        
        Args:
            texts: List of texts to clean
            batch_size: Number of texts to process at once
            
        Returns:
            List of cleaned texts
        """
        cleaned_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [self.clean(text) for text in batch]
            cleaned_texts.extend(cleaned_batch)
            
        return cleaned_texts
    
    @staticmethod
    @lru_cache(maxsize=128)
    def remove_stopwords(text: str, stopwords: frozenset) -> str:
        """
        Remove stopwords from text with caching for performance.
        
        Args:
            text: Input text
            stopwords: Frozen set of stopwords
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in stopwords]
        return ' '.join(filtered_words)


class MultilingualTextCleaner(TextCleaner):
    """Extended text cleaner with multilingual support."""
    
    def __init__(self, language: str = 'en', **kwargs):
        """
        Initialize multilingual text cleaner.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            **kwargs: Additional arguments for parent TextCleaner
        """
        super().__init__(**kwargs)
        self.language = language
        self._load_language_specific_patterns()
        
    def _load_language_specific_patterns(self):
        """Load language-specific cleaning patterns."""
        # Language-specific patterns can be added here
        if self.language == 'en':
            # English-specific patterns
            self.contraction_patterns = {
                r"won't": "will not",
                r"can't": "cannot",
                r"n't": " not",
                r"'re": " are",
                r"'ve": " have",
                r"'ll": " will",
                r"'d": " would",
                r"'m": " am"
            }
        else:
            self.contraction_patterns = {}
            
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text (e.g., "don't" -> "do not").
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for pattern, replacement in self.contraction_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
        
    def clean(self, text: str) -> str:
        """
        Clean text with language-specific processing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Expand contractions before general cleaning
        if self.language == 'en' and self.contraction_patterns:
            text = self.expand_contractions(text)
            
        # Apply general cleaning
        return super().clean(text)