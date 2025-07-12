# NLP Models Module

This module provides BERT-based sentiment analysis and spaCy-based entity extraction with efficient model management.

## Features

### Sentiment Analysis (`sentiment_analyzer.py`)
- BERT-based sentiment classification (positive/negative/neutral)
- Confidence scores for predictions
- Batch processing support
- Sentiment distribution analysis

### Entity Extraction (`entity_extractor.py`)
- spaCy-based named entity recognition
- Support for multiple entity types (Person, Organization, Location, etc.)
- Entity relationship detection
- Context extraction around entities

### Model Management (`model_manager.py`)
- Automatic model loading and caching
- LRU eviction for memory management
- Model versioning support
- Thread-safe operations
- Batch prediction convenience methods

## Installation

```bash
# Install required dependencies
pip install -r requirements_models.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from src.models import ModelManager

# Initialize manager
manager = ModelManager()

# Sentiment analysis
sentiment_result = manager.predict("sentiment", "I love this product!")
print(f"Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.2f})")

# Entity extraction
entity_result = manager.predict("entity", "Apple Inc. was founded by Steve Jobs.")
for entity in entity_result['entities']:
    print(f"Found: {entity['text']} ({entity['type']})")
```

### Advanced Usage

```python
# Batch processing
texts = ["Great service!", "Terrible experience.", "It's okay."]
results = manager.predict_batch("sentiment", texts, batch_size=32)

# Custom model versions
manager.load_model("sentiment", model_name="custom-bert", version="2.0.0")

# Entity type filtering
entities = entity_model.extract_entity_types(texts, ["Person", "Organization"])
```

## Model Configuration

Models can be configured through the ModelManager:

```python
manager.register_model_config("sentiment", {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "version": "1.0.0",
    "max_length": 512
})
```

## Performance Optimization

The module implements several optimizations:
- Model caching to avoid reloading
- Batch processing for efficiency
- LRU eviction to manage memory
- Thread-safe operations for concurrent use

## Supported Models

### Sentiment Analysis
- Default: `nlptown/bert-base-multilingual-uncased-sentiment`
- Any HuggingFace sequence classification model

### Entity Extraction
- Default: `en_core_web_sm`
- Any spaCy model with NER support

## Error Handling

The module provides comprehensive error handling:
- Model loading failures
- Invalid input validation
- Memory management
- Thread safety