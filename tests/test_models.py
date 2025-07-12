"""Comprehensive tests for NLP models."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import torch
import numpy as np

from src.models.base_model import BaseModel
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.entity_extractor import EntityExtractor
from src.models.model_manager import ModelManager


class TestBaseModel:
    """Test suite for BaseModel abstract class."""
    
    def test_init(self):
        """Test BaseModel initialization."""
        # Create a concrete implementation for testing
        class ConcreteModel(BaseModel):
            def load_model(self):
                self.model = "loaded"
                self.is_loaded = True
            
            def predict(self, text):
                return {"result": "test"}
            
            def predict_batch(self, texts, batch_size=32):
                return [{"result": "test"} for _ in texts]
        
        model = ConcreteModel("test-model", "1.0.0")
        assert model.model_name == "test-model"
        assert model.version == "1.0.0"
        assert model.model is None
        assert model.is_loaded is False
        assert model.load_timestamp is None
        assert model._config == {}
    
    def test_unload_model(self):
        """Test model unloading."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                self.model = "loaded"
                self.is_loaded = True
                self.load_timestamp = datetime.now()
            
            def predict(self, text):
                return {"result": "test"}
            
            def predict_batch(self, texts, batch_size=32):
                return [{"result": "test"} for _ in texts]
        
        model = ConcreteModel("test-model")
        model.load_model()
        
        assert model.is_loaded is True
        model.unload_model()
        
        assert model.model is None
        assert model.is_loaded is False
        assert model.load_timestamp is None
    
    def test_get_info(self):
        """Test model info retrieval."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                self.model = "loaded"
                self.is_loaded = True
                self.load_timestamp = datetime.now()
                self._config = {"test": "config"}
            
            def predict(self, text):
                return {"result": "test"}
            
            def predict_batch(self, texts, batch_size=32):
                return [{"result": "test"} for _ in texts]
        
        model = ConcreteModel("test-model", "2.0.0")
        model.load_model()
        
        info = model.get_info()
        assert info["name"] == "test-model"
        assert info["version"] == "2.0.0"
        assert info["is_loaded"] is True
        assert info["load_timestamp"] is not None
        assert info["config"] == {"test": "config"}
    
    def test_validate_input_string(self):
        """Test input validation with string."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                pass
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test")
        result = model.validate_input("test text")
        assert result == ["test text"]
    
    def test_validate_input_list(self):
        """Test input validation with list."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                pass
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test")
        texts = ["text1", "text2", "text3"]
        result = model.validate_input(texts)
        assert result == texts
    
    def test_validate_input_invalid_type(self):
        """Test input validation with invalid type."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                pass
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test")
        with pytest.raises(ValueError, match="Input must be a string or list of strings"):
            model.validate_input(123)
    
    def test_validate_input_non_string_in_list(self):
        """Test input validation with non-string in list."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                pass
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test")
        with pytest.raises(ValueError, match="All inputs must be strings"):
            model.validate_input(["text", 123])
    
    def test_validate_input_empty_string(self):
        """Test input validation with empty string."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                pass
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test")
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            model.validate_input("")
    
    def test_repr(self):
        """Test string representation."""
        class ConcreteModel(BaseModel):
            def load_model(self):
                self.is_loaded = True
            def predict(self, text):
                return {}
            def predict_batch(self, texts, batch_size=32):
                return []
        
        model = ConcreteModel("test-model", "1.0.0")
        expected = "ConcreteModel(name='test-model', version='1.0.0', loaded=False)"
        assert repr(model) == expected
        
        model.load_model()
        expected = "ConcreteModel(name='test-model', version='1.0.0', loaded=True)"
        assert repr(model) == expected


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    def test_init_default(self):
        """Test SentimentAnalyzer initialization with defaults."""
        analyzer = SentimentAnalyzer()
        assert analyzer.model_name == "nlptown/bert-base-multilingual-uncased-sentiment"
        assert analyzer.version == "1.0.0"
        assert analyzer.tokenizer is None
        assert analyzer.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert analyzer._config["model_name"] == "nlptown/bert-base-multilingual-uncased-sentiment"
        assert analyzer._config["max_length"] == 512
    
    def test_init_custom(self):
        """Test SentimentAnalyzer initialization with custom parameters."""
        analyzer = SentimentAnalyzer("custom-model", "2.0.0")
        assert analyzer.model_name == "custom-model"
        assert analyzer.version == "2.0.0"
        assert analyzer._config["model_name"] == "custom-model"
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_load_model_success(self, mock_model, mock_tokenizer):
        """Test successful model loading."""
        mock_tokenizer.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        assert analyzer.is_loaded is True
        assert analyzer.load_timestamp is not None
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        mock_model_instance.to.assert_called_once()
        mock_model_instance.eval.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model_failure(self, mock_tokenizer):
        """Test model loading failure."""
        mock_tokenizer.side_effect = Exception("Model not found")
        
        analyzer = SentimentAnalyzer()
        with pytest.raises(Exception):
            analyzer.load_model()
        
        assert analyzer.is_loaded is False
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('torch.no_grad')
    def test_predict_single_text(self, mock_no_grad, mock_model, mock_tokenizer):
        """Test prediction on single text."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_logits = torch.tensor([[0.1, 0.2, 0.7]])  # 3-class logits
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        result = analyzer.predict("I love this product!")
        
        assert isinstance(result, dict)
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert "scores" in result
        assert "model_version" in result
        assert result["text"] == "I love this product!"
        assert result["model_version"] == "1.0.0"
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('torch.no_grad')
    def test_predict_batch_texts(self, mock_no_grad, mock_model, mock_tokenizer):
        """Test prediction on batch of texts."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102], [101, 102]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_logits = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        texts = ["I love this!", "This is terrible."]
        result = analyzer.predict(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert all("sentiment" in r for r in result)
    
    def test_predict_not_loaded(self):
        """Test prediction when model is not loaded."""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            analyzer.predict("test text")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('torch.no_grad')
    def test_predict_five_star_model(self, mock_no_grad, mock_model, mock_tokenizer):
        """Test prediction with 5-star rating model."""
        # Setup mocks for 5-class model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        # 5-class logits: 1-star, 2-star, 3-star, 4-star, 5-star
        mock_logits = torch.tensor([[0.1, 0.1, 0.2, 0.3, 0.3]])
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        result = analyzer.predict("Great product!")
        
        assert "scores" in result
        assert "positive" in result["scores"]
        assert "negative" in result["scores"]
        assert "neutral" in result["scores"]
        # Should aggregate 4-5 stars as positive
        assert result["scores"]["positive"] == pytest.approx(0.6, rel=0.1)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_batch_with_batch_size(self, mock_model, mock_tokenizer):
        """Test batch prediction with custom batch size."""
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        analyzer._predict_batch = Mock(return_value=[{"sentiment": "positive"} for _ in range(5)])
        
        texts = ["text"] * 100
        results = analyzer.predict_batch(texts, batch_size=20)
        
        assert len(results) == 100
        # Should be called 5 times (100/20)
        assert analyzer._predict_batch.call_count == 5
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('torch.no_grad')
    def test_analyze_sentiment_distribution(self, mock_no_grad, mock_model, mock_tokenizer):
        """Test sentiment distribution analysis."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102], [101, 102], [101, 102]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_logits = torch.tensor([
            [0.1, 0.2, 0.7],  # positive
            [0.8, 0.1, 0.1],  # negative
            [0.2, 0.6, 0.2]   # neutral
        ])
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        texts = ["Great!", "Terrible!", "Okay."]
        distribution = analyzer.analyze_sentiment_distribution(texts)
        
        assert "total_texts" in distribution
        assert "sentiment_counts" in distribution
        assert "sentiment_percentages" in distribution
        assert "average_confidence" in distribution
        assert "confidence_std" in distribution
        assert distribution["total_texts"] == 3
        assert isinstance(distribution["sentiment_counts"], dict)
        assert isinstance(distribution["sentiment_percentages"], dict)
    
    @pytest.mark.performance
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_sentiment_analysis_performance(self, mock_model, mock_tokenizer, benchmark_texts, performance_timer):
        """Test sentiment analysis performance."""
        # Mock fast processing
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        analyzer._predict_batch = Mock(return_value=[
            {"sentiment": "positive", "confidence": 0.9} for _ in range(len(benchmark_texts))
        ])
        
        performance_timer.start()
        results = analyzer.predict_batch(benchmark_texts)
        elapsed = performance_timer.stop()
        
        assert elapsed < 5.0  # Should complete within 5 seconds
        assert len(results) == len(benchmark_texts)
        
        # Calculate throughput
        throughput = len(benchmark_texts) / elapsed
        assert throughput > 50  # Should process >50 texts per second


class TestEntityExtractor:
    """Test suite for EntityExtractor class."""
    
    def test_init_default(self):
        """Test EntityExtractor initialization with defaults."""
        extractor = EntityExtractor()
        assert extractor.model_name == "en_core_web_sm"
        assert extractor.version == "1.0.0"
        assert extractor._config["model_name"] == "en_core_web_sm"
        assert extractor._config["confidence_threshold"] == 0.0
        assert extractor._config["merge_entities"] is True
    
    def test_init_custom(self):
        """Test EntityExtractor initialization with custom parameters."""
        extractor = EntityExtractor("en_core_web_lg", "2.0.0")
        assert extractor.model_name == "en_core_web_lg"
        assert extractor.version == "2.0.0"
    
    @patch('spacy.load')
    def test_load_model_success(self, mock_spacy_load):
        """Test successful model loading."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        assert extractor.is_loaded is True
        assert extractor.load_timestamp is not None
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
    
    @patch('spacy.load')
    def test_load_model_failure(self, mock_spacy_load):
        """Test model loading failure."""
        mock_spacy_load.side_effect = Exception("Model not found")
        
        extractor = EntityExtractor()
        with pytest.raises(Exception):
            extractor.load_model()
        
        assert extractor.is_loaded is False
    
    @patch('spacy.load')
    def test_predict_single_text(self, mock_spacy_load):
        """Test entity extraction on single text."""
        # Setup mock spaCy doc
        mock_entity = Mock()
        mock_entity.text = "Apple Inc."
        mock_entity.label_ = "ORG"
        mock_entity.start_char = 0
        mock_entity.end_char = 10
        
        mock_token = Mock()
        mock_token.text = "Apple"
        mock_token.pos_ = "PROPN"
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity]
        mock_doc.text = "Apple Inc. is a technology company."
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        
        mock_nlp = Mock()
        mock_nlp.pipe.return_value = iter([mock_doc])
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        result = extractor.predict("Apple Inc. is a technology company.")
        
        assert isinstance(result, dict)
        assert "text" in result
        assert "entities" in result
        assert "entity_counts" in result
        assert "total_entities" in result
        assert "tokens" in result
        assert "pos_tags" in result
        assert "model_version" in result
        assert len(result["entities"]) > 0
        assert result["entities"][0]["text"] == "Apple Inc."
        assert result["entities"][0]["type"] == "Organization"
    
    @patch('spacy.load')
    def test_predict_batch_texts(self, mock_spacy_load):
        """Test entity extraction on batch of texts."""
        # Setup mock spaCy docs
        mock_entity1 = Mock()
        mock_entity1.text = "Apple"
        mock_entity1.label_ = "ORG"
        mock_entity1.start_char = 0
        mock_entity1.end_char = 5
        
        mock_entity2 = Mock()
        mock_entity2.text = "Google"
        mock_entity2.label_ = "ORG"
        mock_entity2.start_char = 0
        mock_entity2.end_char = 6
        
        mock_doc1 = Mock()
        mock_doc1.ents = [mock_entity1]
        mock_doc1.text = "Apple Inc."
        mock_doc1.__iter__ = Mock(return_value=iter([]))
        
        mock_doc2 = Mock()
        mock_doc2.ents = [mock_entity2]
        mock_doc2.text = "Google LLC."
        mock_doc2.__iter__ = Mock(return_value=iter([]))
        
        mock_nlp = Mock()
        mock_nlp.pipe.return_value = iter([mock_doc1, mock_doc2])
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        texts = ["Apple Inc.", "Google LLC."]
        result = extractor.predict(texts)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert all("entities" in r for r in result)
    
    def test_predict_not_loaded(self):
        """Test prediction when model is not loaded."""
        extractor = EntityExtractor()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            extractor.predict("test text")
    
    @patch('spacy.load')
    def test_predict_batch_with_batch_size(self, mock_spacy_load):
        """Test batch prediction with custom batch size."""
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.__iter__ = Mock(return_value=iter([]))
        
        mock_nlp = Mock()
        mock_nlp.pipe.return_value = iter([mock_doc] * 100)
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        texts = ["text"] * 100
        results = extractor.predict_batch(texts, batch_size=20)
        
        assert len(results) == 100
        mock_nlp.pipe.assert_called_once_with(texts, batch_size=20)
    
    @patch('spacy.load')
    def test_extract_entity_types(self, mock_spacy_load):
        """Test extraction of specific entity types."""
        extractor = EntityExtractor()
        extractor.load_model()
        
        # Mock the predict_batch method
        mock_results = [
            {
                "entities": [
                    {"text": "Apple", "type": "Organization"},
                    {"text": "California", "type": "Location"},
                ]
            },
            {
                "entities": [
                    {"text": "Google", "type": "Organization"},
                    {"text": "John Doe", "type": "Person"},
                ]
            }
        ]
        extractor.predict_batch = Mock(return_value=mock_results)
        
        texts = ["Apple in California", "Google and John Doe"]
        entity_types = extractor.extract_entity_types(texts, ["Organization"])
        
        assert "Organization" in entity_types
        assert set(entity_types["Organization"]) == {"Apple", "Google"}
        assert "Person" not in entity_types  # Filtered out
    
    @patch('spacy.load')
    def test_find_entity_relationships(self, mock_spacy_load):
        """Test entity relationship finding."""
        # Setup mock entities in a sentence
        mock_entity1 = Mock()
        mock_entity1.text = "John"
        mock_entity1.label_ = "PERSON"
        mock_entity1.start = 0
        mock_entity1.end = 1
        
        mock_entity2 = Mock()
        mock_entity2.text = "Apple"
        mock_entity2.label_ = "ORG"
        mock_entity2.start = 3
        mock_entity2.end = 4
        
        # Mock sentence with entities
        mock_sent = Mock()
        mock_sent.text = "John works at Apple"
        mock_sent.start = 0
        mock_sent.end = 5
        
        # Mock token between entities
        mock_verb_token = Mock()
        mock_verb_token.text = "works"
        mock_verb_token.pos_ = "VERB"
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity1, mock_entity2]
        mock_doc.sents = [mock_sent]
        mock_doc.__getitem__ = Mock(return_value=[mock_verb_token])
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        relationships = extractor.find_entity_relationships("John works at Apple")
        
        assert isinstance(relationships, list)
        # Should find relationship between John and Apple
    
    @patch('spacy.load')
    def test_get_entity_context(self, mock_spacy_load):
        """Test getting entity context."""
        extractor = EntityExtractor()
        extractor.load_model()
        
        # Mock the predict method
        mock_result = {
            "entities": [
                {
                    "text": "Apple",
                    "type": "Organization",
                    "start": 0,
                    "end": 5
                }
            ]
        }
        extractor.predict = Mock(return_value=mock_result)
        
        text = "Apple is a great company"
        contexts = extractor.get_entity_context(text, window_size=10)
        
        assert isinstance(contexts, list)
        assert len(contexts) == 1
        assert "entity" in contexts[0]
        assert "context" in contexts[0]
        assert "entity_position" in contexts[0]
    
    @pytest.mark.performance
    @patch('spacy.load')
    def test_entity_extraction_performance(self, mock_spacy_load, benchmark_texts, performance_timer):
        """Test entity extraction performance."""
        # Mock fast processing
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.__iter__ = Mock(return_value=iter([]))
        
        mock_nlp = Mock()
        mock_nlp.pipe.return_value = iter([mock_doc] * len(benchmark_texts))
        mock_spacy_load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor.load_model()
        
        performance_timer.start()
        results = extractor.predict_batch(benchmark_texts)
        elapsed = performance_timer.stop()
        
        assert elapsed < 10.0  # Should complete within 10 seconds
        assert len(results) == len(benchmark_texts)
        
        # Calculate throughput
        throughput = len(benchmark_texts) / elapsed
        assert throughput > 20  # Should process >20 texts per second


class TestModelManager:
    """Test suite for ModelManager class."""
    
    def test_init(self, temp_dir):
        """Test ModelManager initialization."""
        manager = ModelManager(cache_dir=str(temp_dir))
        assert manager.cache_dir == str(temp_dir)
        assert manager._models == {}
    
    def test_get_model_new(self, temp_dir):
        """Test getting a new model instance."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Mock model class
        mock_model_class = Mock()
        mock_instance = Mock()
        mock_instance.is_loaded = False
        mock_model_class.return_value = mock_instance
        
        model = manager.get_model("test_model", mock_model_class, model_name="test")
        
        assert model == mock_instance
        assert "test_model" in manager._models
        mock_model_class.assert_called_once_with(model_name="test")
    
    def test_get_model_cached(self, temp_dir):
        """Test getting a cached model instance."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Create mock cached model
        mock_cached_model = Mock()
        mock_cached_model.is_loaded = True
        manager._models["test_model"] = mock_cached_model
        
        mock_model_class = Mock()
        
        model = manager.get_model("test_model", mock_model_class)
        
        assert model == mock_cached_model
        # Should not create new instance
        mock_model_class.assert_not_called()
    
    def test_get_model_reload_if_not_loaded(self, temp_dir):
        """Test reloading model if not loaded."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Create mock cached model that's not loaded
        mock_cached_model = Mock()
        mock_cached_model.is_loaded = False
        manager._models["test_model"] = mock_cached_model
        
        mock_model_class = Mock()
        mock_new_instance = Mock()
        mock_model_class.return_value = mock_new_instance
        
        model = manager.get_model("test_model", mock_model_class)
        
        assert model == mock_new_instance
        mock_model_class.assert_called_once()
    
    def test_unload_model(self, temp_dir):
        """Test unloading a model."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Add mock model to cache
        mock_model = Mock()
        manager._models["test_model"] = mock_model
        
        manager.unload_model("test_model")
        
        mock_model.unload_model.assert_called_once()
        assert "test_model" not in manager._models
    
    def test_unload_model_not_found(self, temp_dir):
        """Test unloading a model that doesn't exist."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Should not raise error
        manager.unload_model("nonexistent_model")
    
    def test_clear_cache(self, temp_dir):
        """Test clearing the model cache."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Add mock models to cache
        mock_model1 = Mock()
        mock_model2 = Mock()
        manager._models["model1"] = mock_model1
        manager._models["model2"] = mock_model2
        
        manager.clear_cache()
        
        mock_model1.unload_model.assert_called_once()
        mock_model2.unload_model.assert_called_once()
        assert manager._models == {}
    
    def test_list_models(self, temp_dir):
        """Test listing loaded models."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Add mock models to cache
        mock_model1 = Mock()
        mock_model1.get_info.return_value = {"name": "model1", "loaded": True}
        mock_model2 = Mock()
        mock_model2.get_info.return_value = {"name": "model2", "loaded": False}
        
        manager._models["model1"] = mock_model1
        manager._models["model2"] = mock_model2
        
        models = manager.list_models()
        
        assert len(models) == 2
        assert models["model1"]["name"] == "model1"
        assert models["model2"]["name"] == "model2"
    
    def test_get_model_info(self, temp_dir):
        """Test getting model information."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        mock_model = Mock()
        mock_info = {"name": "test_model", "version": "1.0.0"}
        mock_model.get_info.return_value = mock_info
        manager._models["test_model"] = mock_model
        
        info = manager.get_model_info("test_model")
        
        assert info == mock_info
        mock_model.get_info.assert_called_once()
    
    def test_get_model_info_not_found(self, temp_dir):
        """Test getting info for non-existent model."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        info = manager.get_model_info("nonexistent_model")
        
        assert info is None


# Integration tests
class TestModelsIntegration:
    """Integration tests for model components."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('spacy.load')
    def test_sentiment_and_entity_integration(self, mock_spacy, mock_model, mock_tokenizer, entity_rich_texts):
        """Test sentiment analyzer and entity extractor together."""
        # Setup sentiment analyzer mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_logits = torch.tensor([[0.1, 0.2, 0.7]])
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        # Setup entity extractor mocks
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_nlp.pipe.return_value = iter([mock_doc])
        mock_spacy.return_value = mock_nlp
        
        # Test integration
        sentiment_analyzer = SentimentAnalyzer()
        entity_extractor = EntityExtractor()
        
        sentiment_analyzer.load_model()
        entity_extractor.load_model()
        
        for text in entity_rich_texts[:3]:  # Test subset for speed
            sentiment_result = sentiment_analyzer.predict(text)
            entity_result = entity_extractor.predict(text)
            
            assert isinstance(sentiment_result, dict)
            assert isinstance(entity_result, dict)
            assert "sentiment" in sentiment_result
            assert "entities" in entity_result
    
    def test_model_manager_integration(self, temp_dir):
        """Test ModelManager with actual model classes."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # Test with mock classes that behave like real models
        class MockSentimentAnalyzer(BaseModel):
            def load_model(self):
                self.model = "sentiment_model"
                self.is_loaded = True
                self.load_timestamp = datetime.now()
            
            def predict(self, text):
                return {"sentiment": "positive", "confidence": 0.85}
            
            def predict_batch(self, texts, batch_size=32):
                return [self.predict(text) for text in texts]
        
        class MockEntityExtractor(BaseModel):
            def load_model(self):
                self.model = "entity_model"
                self.is_loaded = True
                self.load_timestamp = datetime.now()
            
            def predict(self, text):
                return {"entities": []}
            
            def predict_batch(self, texts, batch_size=32):
                return [self.predict(text) for text in texts]
        
        # Get models through manager
        sentiment_model = manager.get_model("sentiment", MockSentimentAnalyzer)
        entity_model = manager.get_model("entity", MockEntityExtractor)
        
        assert sentiment_model.is_loaded is True
        assert entity_model.is_loaded is True
        
        # Test predictions
        sentiment_result = sentiment_model.predict("test text")
        entity_result = entity_model.predict("test text")
        
        assert sentiment_result["sentiment"] == "positive"
        assert isinstance(entity_result["entities"], list)
        
        # Test caching
        same_sentiment_model = manager.get_model("sentiment", MockSentimentAnalyzer)
        assert same_sentiment_model is sentiment_model
        
        # Test unloading
        manager.unload_model("sentiment")
        assert "sentiment" not in manager._models
        
        # Test clear cache
        manager.clear_cache()
        assert len(manager._models) == 0


# Error handling tests
class TestModelsErrorHandling:
    """Test error handling in model components."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_sentiment_analyzer_tokenizer_error(self, mock_tokenizer):
        """Test sentiment analyzer handling tokenizer errors."""
        mock_tokenizer.side_effect = Exception("Tokenizer error")
        
        analyzer = SentimentAnalyzer()
        with pytest.raises(Exception):
            analyzer.load_model()
    
    @patch('spacy.load')
    def test_entity_extractor_model_error(self, mock_spacy):
        """Test entity extractor handling model errors."""
        mock_spacy.side_effect = Exception("Model not found")
        
        extractor = EntityExtractor()
        with pytest.raises(Exception):
            extractor.load_model()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_sentiment_analyzer_prediction_error(self, mock_model, mock_tokenizer):
        """Test sentiment analyzer handling prediction errors."""
        mock_tokenizer.return_value = Mock()
        mock_model_instance = Mock()
        mock_model_instance.side_effect = Exception("Prediction error")
        mock_model.return_value = mock_model_instance
        
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
        
        with pytest.raises(Exception):
            analyzer.predict("test text")
    
    def test_model_manager_invalid_model_class(self, temp_dir):
        """Test ModelManager with invalid model class."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        # This should raise an error when trying to instantiate
        with pytest.raises(TypeError):
            manager.get_model("invalid", str)  # str is not a valid model class


# Performance and stress tests
class TestModelsPerformance:
    """Performance tests for model components."""
    
    @pytest.mark.performance
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_sentiment_analyzer_memory_usage(self, mock_model, mock_tokenizer):
        """Test sentiment analyzer memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Create multiple analyzers
        analyzers = []
        for i in range(10):
            analyzer = SentimentAnalyzer(f"model-{i}")
            analyzer.load_model()
            analyzers.append(analyzer)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024
    
    @pytest.mark.performance
    def test_model_manager_cache_performance(self, temp_dir, performance_timer):
        """Test ModelManager cache performance."""
        manager = ModelManager(cache_dir=str(temp_dir))
        
        class FastMockModel(BaseModel):
            def load_model(self):
                self.model = "loaded"
                self.is_loaded = True
            
            def predict(self, text):
                return {"result": "test"}
            
            def predict_batch(self, texts, batch_size=32):
                return [{"result": "test"} for _ in texts]
        
        # Test cache hit performance
        performance_timer.start()
        for i in range(1000):
            model = manager.get_model("test_model", FastMockModel)
        elapsed = performance_timer.stop()
        
        # Should be very fast for cache hits
        assert elapsed < 1.0  # Less than 1 second for 1000 cache hits
        
        # Should only have created one instance
        assert len(manager._models) == 1