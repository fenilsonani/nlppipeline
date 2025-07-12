"""BERT-based sentiment analysis model."""

from typing import Dict, List, Union, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from datetime import datetime
import numpy as np

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SentimentAnalyzer(BaseModel):
    """BERT-based sentiment analysis model."""
    
    SENTIMENT_LABELS = {
        0: "negative",
        1: "neutral", 
        2: "positive"
    }
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment", 
                 version: str = "1.0.0"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            version: Model version
        """
        super().__init__(model_name, version)
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = {
            "model_name": model_name,
            "device": str(self.device),
            "max_length": 512
        }
        
    def load_model(self) -> None:
        """Load BERT model and tokenizer."""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.load_timestamp = datetime.now()
            
            logger.info(f"Sentiment model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict sentiment for input text.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Sentiment predictions with confidence scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        texts = self.validate_input(text)
        is_single = isinstance(text, str)
        
        # Make predictions
        results = self._predict_batch(texts)
        
        return results[0] if is_single else results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        texts = self.validate_input(texts)
        
        # Process in batches
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
            
        return results
    
    def _predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Internal method to predict sentiment for a batch.
        
        Args:
            texts: List of texts
            
        Returns:
            List of predictions
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._config["max_length"],
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Format results
        results = []
        for idx, probs in enumerate(probabilities):
            # Map 5-star rating to 3-class sentiment
            # 1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive
            if len(probs) == 5:  # 5-star rating model
                neg_prob = probs[0] + probs[1]  # 1-2 stars
                neu_prob = probs[2]  # 3 stars
                pos_prob = probs[3] + probs[4]  # 4-5 stars
                
                sentiment_probs = {
                    "negative": float(neg_prob),
                    "neutral": float(neu_prob),
                    "positive": float(pos_prob)
                }
            elif len(probs) == 2:  # Binary classification (positive/negative)
                # Map to 3-class with no neutral
                sentiment_probs = {
                    "negative": float(probs[0]),
                    "neutral": 0.0,
                    "positive": float(probs[1])
                }
            else:  # 3-class model
                sentiment_probs = {
                    label: float(probs[i]) 
                    for i, label in self.SENTIMENT_LABELS.items()
                }
            
            # Get predicted sentiment
            predicted_sentiment = max(sentiment_probs, key=sentiment_probs.get)
            confidence = sentiment_probs[predicted_sentiment]
            
            results.append({
                "text": texts[idx],
                "sentiment": predicted_sentiment,
                "confidence": confidence,
                "scores": sentiment_probs,
                "model_version": self.version
            })
            
        return results
    
    def analyze_sentiment_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment distribution across multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Sentiment distribution statistics
        """
        predictions = self.predict_batch(texts)
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        confidence_scores = []
        
        for pred in predictions:
            sentiment_counts[pred["sentiment"]] += 1
            confidence_scores.append(pred["confidence"])
        
        # Calculate percentages
        total = len(predictions)
        sentiment_percentages = {
            sentiment: (count / total) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        return {
            "total_texts": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "average_confidence": float(np.mean(confidence_scores)),
            "confidence_std": float(np.std(confidence_scores))
        }