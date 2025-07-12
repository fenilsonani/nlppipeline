"""Pytest configuration and shared fixtures for NLP pipeline tests."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any
import time
import json

# Test data fixtures
@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a great product! I love how it works seamlessly.",
        "The customer service was terrible and very disappointing.",
        "Apple Inc. announced new products yesterday in Cupertino, California.",
        "John Smith works at Microsoft Corporation in Seattle.",
        "The weather is nice today.",
        "",  # Empty text
        "   ",  # Whitespace only
        "Check out https://example.com for more info! Email me at test@example.com",
        "Price: $99.99, Date: January 15, 2024, Person: Dr. Jane Doe",
        "Mixed content with numbers 123 and special chars @#$%!"
    ]

@pytest.fixture
def sample_text():
    """Single sample text for testing."""
    return "Apple Inc. is planning to release new iPhone models in September 2024. The CEO Tim Cook will present at the event in Cupertino."

@pytest.fixture
def multilingual_texts():
    """Multilingual sample texts."""
    return [
        "Hello world!",  # English
        "Hola mundo!",   # Spanish
        "Bonjour le monde!",  # French
        "Hallo Welt!",   # German
        "こんにちは世界！"  # Japanese
    ]

@pytest.fixture
def noisy_texts():
    """Texts with various noise patterns for cleaning tests."""
    return [
        "Check out https://example.com/page?param=value&other=123",
        "Contact me at user@domain.com or call 555-123-4567",
        "Price is $19.99!!! What a DEAL!!!",
        "can't won't didn't I'm you're",
        "Text with    extra     whitespace   everywhere",
        "MiXeD cAsE tExT WiTh PuNcTuAtIoN!!!",
        "Numbers: 123, 456.789, 0.001, 50%",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        "Unicode: café, naïve, résumé, Zürich",
    ]

@pytest.fixture
def entity_rich_texts():
    """Texts with various entity types for NER testing."""
    return [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
        "The meeting is scheduled for Monday, December 25th at 3:00 PM PST.",
        "Amazon reported revenue of $469.8 billion in 2021, a 22% increase.",
        "Dr. Sarah Johnson works at Stanford University in Palo Alto.",
        "The Eiffel Tower in Paris, France was completed in 1889.",
        "Google's headquarters are located at 1600 Amphitheatre Parkway, Mountain View.",
        "The book '1984' by George Orwell was published in June 1949.",
    ]

@pytest.fixture
def sentiment_labeled_texts():
    """Texts with known sentiment labels for testing."""
    return [
        {"text": "I absolutely love this product! It's amazing!", "sentiment": "positive"},
        {"text": "This is the worst experience I've ever had.", "sentiment": "negative"},
        {"text": "The weather is okay today.", "sentiment": "neutral"},
        {"text": "Fantastic customer service and quick delivery!", "sentiment": "positive"},
        {"text": "Terrible quality and poor customer support.", "sentiment": "negative"},
        {"text": "The product works as expected.", "sentiment": "neutral"},
    ]

# Mock fixtures
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = Mock()
    mock.tokenize.return_value = Mock(
        tokens=['mock', 'tokens'],
        token_ids=[1, 2, 3],
        attention_mask=[1, 1, 1]
    )
    mock.vocab_size = 30000
    mock.special_tokens = {
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'mask_token': '[MASK]'
    }
    return mock

@pytest.fixture
def mock_sentiment_model():
    """Mock sentiment analyzer for testing."""
    mock = Mock()
    mock.is_loaded = True
    mock.model_name = "mock-sentiment-model"
    mock.version = "1.0.0"
    
    def mock_predict(text):
        if isinstance(text, str):
            return {
                "text": text,
                "sentiment": "positive",
                "confidence": 0.85,
                "scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05},
                "model_version": "1.0.0"
            }
        return [mock_predict(t) for t in text]
    
    mock.predict.side_effect = mock_predict
    mock.predict_batch.side_effect = mock_predict
    return mock

@pytest.fixture
def mock_entity_model():
    """Mock entity extractor for testing."""
    mock = Mock()
    mock.is_loaded = True
    mock.model_name = "mock-entity-model"
    mock.version = "1.0.0"
    
    def mock_predict(text):
        if isinstance(text, str):
            return {
                "text": text,
                "entities": [
                    {
                        "text": "Apple Inc.",
                        "type": "Organization",
                        "label": "ORG",
                        "start": 0,
                        "end": 10,
                        "confidence": 0.95
                    }
                ],
                "entity_counts": {"Organization": 1},
                "total_entities": 1,
                "model_version": "1.0.0"
            }
        return [mock_predict(t) for t in text]
    
    mock.predict.side_effect = mock_predict
    mock.predict_batch.side_effect = mock_predict
    return mock

@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.model.bert_model_path = "bert-base-uncased"
    config.model.spacy_model = "en_core_web_sm"
    config.model.model_cache_dir = "/tmp/models"
    config.model.batch_size = 32
    config.kafka.bootstrap_servers = "localhost:9092"
    config.kafka.topic = "nlp-input"
    config.kafka.group_id = "nlp-consumer"
    config.storage.output_path = Path("/tmp/output")
    config.to_dict.return_value = {"test": "config"}
    return config

# File system fixtures
@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_json_file(temp_dir):
    """Sample JSON file for testing."""
    data = [
        {"document_id": "doc1", "text": "This is a positive review!"},
        {"document_id": "doc2", "text": "This product is terrible."},
        {"document_id": "doc3", "text": "Average product quality."}
    ]
    file_path = temp_dir / "sample.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def sample_csv_file(temp_dir):
    """Sample CSV file for testing."""
    content = """document_id,text
doc1,"This is a positive review!"
doc2,"This product is terrible."
doc3,"Average product quality."
"""
    file_path = temp_dir / "sample.csv"
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

@pytest.fixture
def large_text_batch():
    """Large batch of texts for performance testing."""
    base_texts = [
        "This is a sample text for performance testing.",
        "Another sample text with different content.",
        "Performance testing requires various text samples.",
        "Testing the processing speed of the pipeline.",
        "Large batches help identify bottlenecks."
    ]
    # Create 1000 texts by repeating base texts
    return base_texts * 200

# Async testing fixtures
@pytest.fixture
def event_loop():
    """Event loop for async testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def async_mock_pipeline():
    """Async mock pipeline for testing."""
    pipeline = Mock()
    
    async def mock_process_document(doc):
        await asyncio.sleep(0.001)  # Simulate processing time
        return Mock(
            document_id="test_doc",
            text=doc if isinstance(doc, str) else doc.get('text', ''),
            sentiment="positive",
            sentiment_confidence=0.85,
            entities=[],
            processing_time=0.001,
            timestamp=time.time()
        )
    
    async def mock_process_batch(docs):
        tasks = [mock_process_document(doc) for doc in docs]
        return await asyncio.gather(*tasks)
    
    pipeline.process_document = mock_process_document
    pipeline.process_batch = mock_process_batch
    pipeline.documents_processed = 0
    pipeline.total_processing_time = 0.0
    
    return pipeline

# Error testing fixtures
@pytest.fixture
def error_prone_texts():
    """Texts that might cause processing errors."""
    return [
        None,  # None value
        123,   # Non-string type
        "",    # Empty string
        " " * 10000,  # Very long whitespace
        "a" * 100000,  # Very long text
        "\x00\x01\x02",  # Control characters
        "🚀🌟💫🎉🔥",  # Only emojis
    ]

# Model-specific fixtures
@pytest.fixture
def bert_tokenizer_config():
    """Configuration for BERT tokenizer testing."""
    return {
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt"
    }

@pytest.fixture
def spacy_model_config():
    """Configuration for spaCy model testing."""
    return {
        "model_name": "en_core_web_sm",
        "confidence_threshold": 0.0,
        "merge_entities": True
    }

# Benchmark fixtures
@pytest.fixture
def benchmark_texts():
    """Standard benchmark texts for consistent testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Apple Inc. is a technology company based in Cupertino, California.",
        "I really enjoyed the movie and would recommend it to others.",
        "The customer service was poor and needs improvement.",
        "Today is January 15, 2024, and the weather is sunny."
    ] * 100  # 500 texts total

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Add any necessary cleanup here
    pass

# Pytest configuration
def pytest_configure(config):
    """Pytest configuration setup."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )

# Pytest collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "large_batch" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "test_pipeline" in item.nodeid and "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default)
        if not any(marker.name in ["slow", "integration"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)