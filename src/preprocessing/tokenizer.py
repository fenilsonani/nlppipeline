"""
Text tokenization utilities for NLP preprocessing.
"""

import re
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class TokenizationOutput:
    """Container for tokenization results."""
    tokens: List[str]
    token_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    special_tokens_mask: Optional[List[int]] = None


class BasicTokenizer:
    """Basic text tokenizer with configurable options."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        language: str = 'english',
        min_token_length: int = 1,
        max_token_length: Optional[int] = None
    ):
        """
        Initialize basic tokenizer.
        
        Args:
            lowercase: Convert tokens to lowercase
            remove_stopwords: Remove stopwords from tokens
            language: Language for stopwords
            min_token_length: Minimum token length to keep
            max_token_length: Maximum token length to keep
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Load stopwords if needed
        if self.remove_stopwords:
            try:
                self.stopwords = set(stopwords.words(language))
            except:
                self.stopwords = set()
                print(f"Warning: Could not load stopwords for language '{language}'")
    
    def tokenize(self, text: str) -> TokenizationOutput:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            TokenizationOutput with tokens
        """
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        # Apply lowercase
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        # Remove stopwords
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Filter by length
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        if self.max_token_length:
            tokens = [t for t in tokens if len(t) <= self.max_token_length]
        
        return TokenizationOutput(tokens=tokens)
    
    def tokenize_batch(self, texts: List[str]) -> List[TokenizationOutput]:
        """
        Tokenize multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of TokenizationOutput objects
        """
        return [self.tokenize(text) for text in texts]
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)


class BERTTokenizer:
    """Tokenizer for BERT and other transformer models."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ):
        """
        Initialize BERT tokenizer.
        
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_tensors: Return type ('pt' for PyTorch, 'tf' for TensorFlow)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False
    ) -> TokenizationOutput:
        """
        Tokenize text for BERT model.
        
        Args:
            text: Input text
            add_special_tokens: Add [CLS] and [SEP] tokens
            return_offsets_mapping: Return character offsets
            
        Returns:
            TokenizationOutput with tokens and IDs
        """
        # Tokenize with the pretrained tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_offsets_mapping=return_offsets_mapping,
            return_tensors=self.return_tensors
        )
        
        # Convert to lists if tensors
        if self.return_tensors:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            output = TokenizationOutput(
                tokens=tokens,
                token_ids=encoding['input_ids'][0].tolist(),
                attention_mask=encoding['attention_mask'][0].tolist()
            )
        else:
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            output = TokenizationOutput(
                tokens=tokens,
                token_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
        
        # Add token type IDs if available
        if 'token_type_ids' in encoding:
            output.token_type_ids = encoding['token_type_ids']
        
        return output
    
    def tokenize_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[TokenizationOutput]:
        """
        Tokenize multiple texts efficiently.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            **kwargs: Additional arguments for tokenization
            
        Returns:
            List of TokenizationOutput objects
        """
        all_outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Batch tokenization
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=kwargs.get('add_special_tokens', True),
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=None  # Return lists for batch processing
            )
            
            # Create output objects for each text
            for idx in range(len(batch_texts)):
                tokens = self.tokenizer.convert_ids_to_tokens(encodings['input_ids'][idx])
                output = TokenizationOutput(
                    tokens=tokens,
                    token_ids=encodings['input_ids'][idx],
                    attention_mask=encodings['attention_mask'][idx]
                )
                
                if 'token_type_ids' in encodings:
                    output.token_type_ids = encodings['token_type_ids'][idx]
                
                all_outputs.append(output)
        
        return all_outputs
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def special_tokens(self) -> Dict[str, str]:
        """Get special tokens."""
        return {
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'pad_token': self.tokenizer.pad_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': self.tokenizer.mask_token
        }


class SubwordTokenizer:
    """Tokenizer using subword tokenization (BPE, WordPiece, etc.)."""
    
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2):
        """
        Initialize subword tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for vocabulary
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        
    def train(self, texts: List[str]):
        """
        Train tokenizer on corpus.
        
        Args:
            texts: Training corpus
        """
        # This is a placeholder - actual implementation would use
        # tokenizers library or similar for BPE/WordPiece training
        raise NotImplementedError("Subword tokenizer training not implemented")
    
    def tokenize(self, text: str) -> TokenizationOutput:
        """
        Tokenize text using subword tokenization.
        
        Args:
            text: Input text
            
        Returns:
            TokenizationOutput with subword tokens
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Placeholder implementation
        tokens = text.split()  # Simple split for now
        return TokenizationOutput(tokens=tokens)