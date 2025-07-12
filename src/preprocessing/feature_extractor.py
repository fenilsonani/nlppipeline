"""
Feature extraction utilities for ML models.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import torch
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import lru_cache


@dataclass
class FeatureOutput:
    """Container for extracted features."""
    features: np.ndarray
    feature_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class TfidfFeatureExtractor:
    """TF-IDF based feature extraction."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        norm: str = 'l2'
    ):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Use inverse document frequency
            sublinear_tf: Apply sublinear TF scaling
            norm: Normalization method
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            norm=norm
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer on texts.
        
        Args:
            texts: List of texts to fit on
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def extract(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            FeatureOutput with TF-IDF features
        """
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        features = self.vectorizer.transform(texts).toarray()
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        return FeatureOutput(
            features=features,
            feature_names=feature_names,
            metadata={'type': 'tfidf', 'shape': features.shape}
        )
    
    def fit_extract(self, texts: List[str]) -> FeatureOutput:
        """
        Fit and extract features in one step.
        
        Args:
            texts: List of texts
            
        Returns:
            FeatureOutput with TF-IDF features
        """
        features = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        return FeatureOutput(
            features=features,
            feature_names=feature_names,
            metadata={'type': 'tfidf', 'shape': features.shape}
        )


class BERTFeatureExtractor:
    """BERT-based feature extraction for dense representations."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        pooling_strategy: str = 'cls',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize BERT feature extractor.
        
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max')
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
    
    @torch.no_grad()
    def extract(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """
        Extract BERT features from texts.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            FeatureOutput with BERT embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Apply pooling strategy
            if self.pooling_strategy == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.pooling_strategy == 'mean':
                mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            elif self.pooling_strategy == 'max':
                embeddings = outputs.last_hidden_state.max(1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        features = np.vstack(all_embeddings)
        
        return FeatureOutput(
            features=features,
            metadata={
                'type': 'bert',
                'model': self.model_name,
                'pooling': self.pooling_strategy,
                'shape': features.shape,
                'embedding_dim': self.embedding_dim
            }
        )
    
    @lru_cache(maxsize=1000)
    def _extract_cached(self, text_hash: str) -> np.ndarray:
        """
        Extract features with caching for repeated texts.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Feature vector
        """
        # This is called by extract_with_cache
        pass
    
    def extract_with_cache(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """
        Extract features with caching for performance.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            FeatureOutput with cached extraction
        """
        if isinstance(texts, str):
            texts = [texts]
        
        features = []
        for text in texts:
            # Create hash of text for caching
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache first
            if text_hash in self._extract_cached.cache_info():
                feature = self._extract_cached(text_hash)
            else:
                # Extract and cache
                output = self.extract(text)
                feature = output.features[0]
                # Update cache manually
                self._extract_cached.cache_clear()
                self._extract_cached(text_hash)
            
            features.append(feature)
        
        return FeatureOutput(
            features=np.vstack(features),
            metadata={'type': 'bert', 'cached': True}
        )


class CountVectorFeatureExtractor:
    """Count-based feature extraction (Bag of Words)."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 1),
        binary: bool = False,
        min_df: Union[int, float] = 1
    ):
        """
        Initialize count vector feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams
            binary: Whether to use binary features
            min_df: Minimum document frequency
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            binary=binary,
            min_df=min_df
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the count vectorizer."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def extract(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """Extract count features."""
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        features = self.vectorizer.transform(texts).toarray()
        feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        return FeatureOutput(
            features=features,
            feature_names=feature_names,
            metadata={'type': 'count', 'shape': features.shape}
        )


class TopicFeatureExtractor:
    """Topic modeling based feature extraction using LDA."""
    
    def __init__(
        self,
        n_topics: int = 50,
        max_features: int = 10000,
        learning_method: str = 'batch',
        random_state: int = 42
    ):
        """
        Initialize topic feature extractor.
        
        Args:
            n_topics: Number of topics
            max_features: Maximum vocabulary size
            learning_method: Learning method for LDA
            random_state: Random seed
        """
        self.n_topics = n_topics
        
        # Count vectorizer for LDA
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        
        # LDA model
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method=learning_method,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the topic model."""
        # Vectorize texts
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit LDA
        self.lda.fit(doc_term_matrix)
        self.is_fitted = True
    
    def extract(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """Extract topic features."""
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform to document-term matrix
        doc_term_matrix = self.vectorizer.transform(texts)
        
        # Get topic distributions
        features = self.lda.transform(doc_term_matrix)
        
        return FeatureOutput(
            features=features,
            feature_names=[f'topic_{i}' for i in range(self.n_topics)],
            metadata={'type': 'lda', 'n_topics': self.n_topics, 'shape': features.shape}
        )
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic ID to top words
        """
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics


class EmbeddingFeatureExtractor:
    """Generic embedding-based feature extractor."""
    
    def __init__(
        self,
        embedding_type: str = 'word2vec',
        embedding_dim: int = 300,
        aggregation: str = 'mean'
    ):
        """
        Initialize embedding feature extractor.
        
        Args:
            embedding_type: Type of embeddings
            embedding_dim: Dimension of embeddings
            aggregation: How to aggregate word embeddings
        """
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.embeddings = {}
        
    def load_embeddings(self, embedding_path: str):
        """Load pre-trained embeddings from file."""
        # Placeholder for loading embeddings
        warnings.warn("Embedding loading not implemented. Using random embeddings.")
        
    def extract(self, texts: Union[str, List[str]]) -> FeatureOutput:
        """Extract embedding features."""
        if isinstance(texts, str):
            texts = [texts]
        
        features = []
        
        for text in texts:
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                if word in self.embeddings:
                    word_vectors.append(self.embeddings[word])
                else:
                    # Random vector for OOV words
                    word_vectors.append(np.random.randn(self.embedding_dim))
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                
                if self.aggregation == 'mean':
                    doc_vector = np.mean(word_vectors, axis=0)
                elif self.aggregation == 'max':
                    doc_vector = np.max(word_vectors, axis=0)
                elif self.aggregation == 'sum':
                    doc_vector = np.sum(word_vectors, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            else:
                doc_vector = np.zeros(self.embedding_dim)
            
            features.append(doc_vector)
        
        return FeatureOutput(
            features=np.vstack(features),
            metadata={
                'type': self.embedding_type,
                'dim': self.embedding_dim,
                'aggregation': self.aggregation
            }
        )