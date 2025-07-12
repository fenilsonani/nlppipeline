"""Base class for all NLP models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all NLP models."""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            version: Model version string
        """
        self.model_name = model_name
        self.version = version
        self.model = None
        self.is_loaded = False
        self.load_timestamp = None
        self._config = {}
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make predictions on input text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Prediction results as dictionary or list of dictionaries
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Size of each batch for processing
            
        Returns:
            List of prediction results
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        self.model = None
        self.is_loaded = False
        self.load_timestamp = None
        logger.info(f"Model {self.model_name} v{self.version} unloaded")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "name": self.model_name,
            "version": self.version,
            "is_loaded": self.is_loaded,
            "load_timestamp": self.load_timestamp.isoformat() if self.load_timestamp else None,
            "config": self._config
        }
    
    def validate_input(self, text: Union[str, List[str]]) -> List[str]:
        """
        Validate and prepare input text.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            List of validated texts
            
        Raises:
            ValueError: If input is invalid
        """
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            texts = text
        else:
            raise ValueError("Input must be a string or list of strings")
        
        # Validate each text
        validated = []
        for t in texts:
            if not isinstance(t, str):
                raise ValueError(f"All inputs must be strings, got {type(t)}")
            if not t.strip():
                raise ValueError("Input text cannot be empty")
            validated.append(t.strip())
            
        return validated
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', version='{self.version}', loaded={self.is_loaded})"