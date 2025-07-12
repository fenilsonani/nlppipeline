"""Model loading and management."""

from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
import threading
import json
import os
from pathlib import Path
import pickle
import hashlib

from .base_model import BaseModel
from .sentiment_analyzer import SentimentAnalyzer
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading, caching, and versioning of NLP models."""
    
    # Model registry
    MODEL_CLASSES = {
        "sentiment": SentimentAnalyzer,
        "entity": EntityExtractor
    }
    
    def __init__(self, cache_dir: Optional[str] = None, max_models: int = 5):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for model caching
            max_models: Maximum number of models to keep in memory
        """
        self.models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".nlppipeline" / "models"
        self.max_models = max_models
        self._lock = threading.Lock()
        self._access_times: Dict[str, datetime] = {}
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model configurations
        self._load_configs()
    
    def load_model(self, model_type: str, model_name: Optional[str] = None, 
                   version: Optional[str] = None, force_reload: bool = False) -> BaseModel:
        """
        Load a model with caching support.
        
        Args:
            model_type: Type of model (sentiment, entity)
            model_name: Specific model name (optional)
            version: Model version (optional)
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model instance
        """
        # Validate model type
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODEL_CLASSES.keys())}")
        
        # Generate model key
        model_key = self._get_model_key(model_type, model_name, version)
        
        with self._lock:
            # Check if model is already loaded
            if model_key in self.models and not force_reload:
                logger.info(f"Using cached model: {model_key}")
                self._access_times[model_key] = datetime.now()
                return self.models[model_key]
            
            # Check cache capacity
            if len(self.models) >= self.max_models:
                self._evict_least_recently_used()
            
            # Create and load model
            logger.info(f"Loading model: {model_key}")
            model_class = self.MODEL_CLASSES[model_type]
            
            # Get default config if not specified
            config = self._get_model_config(model_type, model_name, version)
            
            # Create model instance
            model = model_class(
                model_name=config.get("model_name", model_name),
                version=config.get("version", version or "1.0.0")
            )
            
            # Try to load from cache first
            if not force_reload and self._load_from_cache(model_key, model):
                logger.info(f"Loaded model from cache: {model_key}")
            else:
                # Load model from source
                model.load_model()
                # Save to cache
                self._save_to_cache(model_key, model)
            
            # Store in memory
            self.models[model_key] = model
            self._access_times[model_key] = datetime.now()
            
            return model
    
    def get_model(self, model_type: str, model_name: Optional[str] = None,
                  version: Optional[str] = None) -> Optional[BaseModel]:
        """
        Get a loaded model if available.
        
        Args:
            model_type: Type of model
            model_name: Specific model name
            version: Model version
            
        Returns:
            Model instance or None
        """
        model_key = self._get_model_key(model_type, model_name, version)
        
        with self._lock:
            if model_key in self.models:
                self._access_times[model_key] = datetime.now()
                return self.models[model_key]
        
        return None
    
    def unload_model(self, model_type: str, model_name: Optional[str] = None,
                     version: Optional[str] = None) -> bool:
        """
        Unload a specific model from memory.
        
        Args:
            model_type: Type of model
            model_name: Specific model name
            version: Model version
            
        Returns:
            True if unloaded, False if not found
        """
        model_key = self._get_model_key(model_type, model_name, version)
        
        with self._lock:
            if model_key in self.models:
                self.models[model_key].unload_model()
                del self.models[model_key]
                del self._access_times[model_key]
                logger.info(f"Unloaded model: {model_key}")
                return True
        
        return False
    
    def unload_all(self) -> None:
        """Unload all models from memory."""
        with self._lock:
            for model_key in list(self.models.keys()):
                self.models[model_key].unload_model()
            
            self.models.clear()
            self._access_times.clear()
            logger.info("All models unloaded")
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List all currently loaded models.
        
        Returns:
            List of model information dictionaries
        """
        with self._lock:
            return [
                {
                    "key": model_key,
                    "type": model_key.split(":")[0],
                    "info": model.get_info(),
                    "last_accessed": self._access_times.get(model_key)
                }
                for model_key, model in self.models.items()
            ]
    
    def predict(self, model_type: str, text: Union[str, List[str]], 
                model_name: Optional[str] = None, version: Optional[str] = None,
                **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convenience method to load model and make predictions.
        
        Args:
            model_type: Type of model
            text: Input text(s)
            model_name: Specific model name
            version: Model version
            **kwargs: Additional arguments for prediction
            
        Returns:
            Prediction results
        """
        # Load or get model
        model = self.get_model(model_type, model_name, version)
        if model is None:
            model = self.load_model(model_type, model_name, version)
        
        # Make predictions
        return model.predict(text, **kwargs)
    
    def predict_batch(self, model_type: str, texts: List[str],
                      model_name: Optional[str] = None, version: Optional[str] = None,
                      batch_size: int = 32, **kwargs) -> List[Dict[str, Any]]:
        """
        Convenience method for batch predictions.
        
        Args:
            model_type: Type of model
            texts: List of input texts
            model_name: Specific model name
            version: Model version
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            List of prediction results
        """
        # Load or get model
        model = self.get_model(model_type, model_name, version)
        if model is None:
            model = self.load_model(model_type, model_name, version)
        
        # Make batch predictions
        return model.predict_batch(texts, batch_size=batch_size, **kwargs)
    
    def register_model_config(self, model_type: str, config: Dict[str, Any]) -> None:
        """
        Register a model configuration.
        
        Args:
            model_type: Type of model
            config: Model configuration
        """
        if model_type not in self.model_configs:
            self.model_configs[model_type] = {}
        
        config_key = f"{config.get('model_name', 'default')}:{config.get('version', '1.0.0')}"
        self.model_configs[model_type][config_key] = config
        
        # Save configs
        self._save_configs()
    
    def _get_model_key(self, model_type: str, model_name: Optional[str], 
                       version: Optional[str]) -> str:
        """Generate unique model key."""
        name = model_name or "default"
        ver = version or "1.0.0"
        return f"{model_type}:{name}:{ver}"
    
    def _get_model_config(self, model_type: str, model_name: Optional[str],
                          version: Optional[str]) -> Dict[str, Any]:
        """Get model configuration."""
        if model_type not in self.model_configs:
            return {}
        
        config_key = f"{model_name or 'default'}:{version or '1.0.0'}"
        return self.model_configs[model_type].get(config_key, {})
    
    def _evict_least_recently_used(self) -> None:
        """Evict least recently used model."""
        if not self.models:
            return
        
        # Find LRU model
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Unload it
        logger.info(f"Evicting LRU model: {lru_key}")
        self.models[lru_key].unload_model()
        del self.models[lru_key]
        del self._access_times[lru_key]
    
    def _get_cache_path(self, model_key: str) -> Path:
        """Get cache file path for model."""
        # Create hash for filename
        key_hash = hashlib.md5(model_key.encode()).hexdigest()[:8]
        return self.cache_dir / f"{model_key.replace(':', '_')}_{key_hash}.pkl"
    
    def _save_to_cache(self, model_key: str, model: BaseModel) -> None:
        """Save model to cache."""
        try:
            cache_path = self._get_cache_path(model_key)
            
            # Note: This is a simplified cache - in reality, you'd need custom
            # serialization for transformer models
            cache_data = {
                "model_key": model_key,
                "model_info": model.get_info(),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved model to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save model to cache: {e}")
    
    def _load_from_cache(self, model_key: str, model: BaseModel) -> bool:
        """Load model from cache."""
        try:
            cache_path = self._get_cache_path(model_key)
            
            if not cache_path.exists():
                return False
            
            # Note: This is simplified - real implementation would need
            # to handle model state restoration
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            
            logger.info(f"Found cached model: {model_key}")
            return False  # For now, always load fresh
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return False
    
    def _load_configs(self) -> None:
        """Load model configurations from file."""
        config_path = self.cache_dir / "model_configs.json"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    self.model_configs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model configs: {e}")
                self.model_configs = {}
        else:
            # Default configurations
            self.model_configs = {
                "sentiment": {
                    "default:1.0.0": {
                        "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
                        "version": "1.0.0"
                    }
                },
                "entity": {
                    "default:1.0.0": {
                        "model_name": "en_core_web_sm",
                        "version": "1.0.0"
                    }
                }
            }
    
    def _save_configs(self) -> None:
        """Save model configurations to file."""
        config_path = self.cache_dir / "model_configs.json"
        
        try:
            with open(config_path, "w") as f:
                json.dump(self.model_configs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model configs: {e}")