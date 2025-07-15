# NLP Pipeline

## 🚀 Enterprise-Grade Natural Language Processing Pipeline

A comprehensive, production-ready NLP pipeline for sentiment analysis and entity extraction with real-time monitoring, streaming capabilities, and enterprise-grade performance optimizations.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance](#performance)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)

## 🎯 Overview

This NLP pipeline is designed to handle enterprise-scale text processing with a focus on:
- **High Performance**: Sub-200ms response times for most operations
- **Scalability**: Horizontal scaling with Kafka streaming and Spark processing
- **Reliability**: Comprehensive error handling, retries, and monitoring
- **Flexibility**: Modular architecture supporting multiple models and processors
- **Production Ready**: Complete with monitoring, alerting, and deployment configurations

### Key Capabilities

- **Sentiment Analysis**: Advanced sentiment classification with confidence scores
- **Entity Extraction**: Named entity recognition with custom entity types
- **Real-time Processing**: Kafka-based streaming for live data ingestion
- **Batch Processing**: Efficient batch processing with Apache Spark
- **Model Management**: Dynamic model loading/unloading with memory optimization
- **Performance Monitoring**: Real-time metrics collection and alerting

## ✨ Features

### Core NLP Features
- Multi-model sentiment analysis (BERT, RoBERTa, custom models)
- Named Entity Recognition (NER) with 18+ entity types
- Text preprocessing and feature extraction
- Batch and streaming processing modes
- Custom model training and fine-tuning

### Enterprise Features
- **Real-time Streaming**: Kafka integration for live data processing
- **Distributed Processing**: Apache Spark for large-scale batch jobs
- **Model Management**: Hot-swappable models with version control
- **Performance Monitoring**: Prometheus metrics and custom dashboards
- **Health Checks**: Comprehensive system health monitoring
- **Alert Management**: Configurable alerts for performance and errors

### Development Features
- Type hints and comprehensive documentation
- Extensive test coverage (>90%)
- Code formatting with Black and isort
- Jupyter notebooks for experimentation
- Docker containerization support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Ingestion     │    │   Processing    │
│                 │    │                 │    │                 │
│ • Kafka Topics  │───▶│ • Stream Reader │───▶│ • Text Cleaner  │
│ • File Uploads  │    │ • Batch Loader  │    │ • Tokenizer     │
│ • API Requests  │    │ • Spark Jobs    │    │ • Feature Ext.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   ML Models     │    │  Postprocessing │
│                 │    │                 │    │                 │
│ • Metrics       │◀───│ • Sentiment     │◀───│ • Aggregation   │
│ • Health Checks │    │ • Entity NER    │    │ • Visualization │
│ • Alerts        │    │ • Custom Models │    │ • Export        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Overview

1. **Ingestion Layer**: Handles data input from multiple sources
2. **Processing Layer**: Text preprocessing and feature extraction
3. **Model Layer**: ML models for sentiment analysis and NER
4. **Postprocessing Layer**: Result aggregation and visualization
5. **Monitoring Layer**: Performance tracking and health monitoring

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Java 8+ (for Spark integration)
- Docker (optional, for containerized deployment)

### Using pip

```bash
# Clone the repository
git clone https://github.com/fenilsonani/nlppipeline.git
cd nlppipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional model dependencies
pip install -r requirements_models.txt

# Install in development mode
pip install -e .
```

### Using pnpm (for web interface)

```bash
pnpm install
```

### Docker Installation

```bash
# Build the Docker image
docker build -t nlp-pipeline .

# Run with Docker Compose
docker-compose up -d
```

## 🚀 Quick Start

### 1. Basic Setup

```python
from src.models import ModelManager

# Initialize the model manager
manager = ModelManager()

# Load models
sentiment_model = manager.load_model("sentiment")
entity_model = manager.load_model("entity")
```

### 2. Sentiment Analysis

```python
# Single text analysis
text = "I love this product! It's absolutely amazing!"
result = sentiment_model.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Scores: {result['scores']}")

# Batch analysis
texts = [
    "Great service!",
    "Terrible experience.",
    "It's okay, nothing special."
]
results = sentiment_model.predict_batch(texts)
```

### 3. Entity Extraction

```python
# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
result = entity_model.predict(text)

print(f"Found {result['total_entities']} entities:")
for entity in result['entities']:
    print(f"- {entity['text']} ({entity['type']})")
```

### 4. Streaming Processing

```python
from src.ingestion import KafkaStreamConsumer

def process_message(message):
    # Process incoming message
    text = message['text']
    sentiment = sentiment_model.predict(text)
    entities = entity_model.predict(text)
    
    # Store or forward results
    print(f"Processed: {text[:50]}...")

# Start streaming consumer
consumer = KafkaStreamConsumer()
consumer.consume_messages(process_message)
```

## 💡 Usage Examples

### Advanced Sentiment Analysis

```python
# Analyze sentiment distribution
texts = ["Great!", "Terrible!", "Okay", "Amazing!", "Bad"]
distribution = sentiment_model.analyze_sentiment_distribution(texts)
print(distribution)
# Output: {'positive': 2, 'negative': 2, 'neutral': 1}

# Get detailed scores for all sentiments
result = sentiment_model.predict("This is interesting", return_all_scores=True)
print(result['all_scores'])
# Output: {'positive': 0.3, 'negative': 0.2, 'neutral': 0.5}
```

### Entity Extraction with Relationships

```python
# Extract specific entity types
text = "Microsoft CEO Satya Nadella met with President Biden in Washington."
entity_types = entity_model.extract_entity_types(text, ["Person", "Organization"])
print(entity_types)

# Find entity relationships
relationships = entity_model.find_entity_relationships(text)
for rel in relationships:
    print(f"{rel['entity1']['text']} --{rel['relation']}--> {rel['entity2']['text']}")
```

### Batch Processing with Spark

```python
from src.ingestion import SparkProcessor

# Initialize Spark processor
processor = SparkProcessor()

# Process large dataset
results = processor.process_batch(
    input_path="data/raw/documents.json",
    output_path="data/processed/results.json",
    models=["sentiment", "entity"]
)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Kafka Configuration
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_GROUP_ID="nlp-pipeline-consumer"
export KAFKA_TOPIC="nlp-input"

# Model Configuration
export MODEL_CACHE_SIZE=5
export MODEL_CACHE_TTL=3600

# Performance Configuration
export BATCH_SIZE=32
export MAX_WORKERS=4
export RESPONSE_TIMEOUT=30
```

### Configuration Files

Create a `config/settings.yaml` file:

```yaml
models:
  sentiment:
    model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: 512
    batch_size: 32
  
  entity:
    model_name: "dbmdz/bert-large-cased-finetuned-conll03-english"
    max_length: 512
    batch_size: 16

kafka:
  bootstrap_servers: "localhost:9092"
  group_id: "nlp-pipeline"
  auto_offset_reset: "latest"

monitoring:
  metrics_interval: 10
  health_check_interval: 30
  alert_thresholds:
    response_time_ms: 200
    error_rate_percent: 5
```

## 📊 Performance

### Benchmarks

| Operation | Avg Response Time | Throughput | Memory Usage |
|-----------|------------------|------------|--------------|
| Sentiment Analysis (single) | 15ms | 67 req/sec | 150MB |
| Sentiment Analysis (batch-32) | 180ms | 180 req/sec | 200MB |
| Entity Extraction (single) | 25ms | 40 req/sec | 180MB |
| Entity Extraction (batch-16) | 200ms | 80 req/sec | 250MB |

### Performance Optimization

- **Model Caching**: Keep frequently used models in memory
- **Batch Processing**: Process multiple texts together for better throughput
- **Async Processing**: Non-blocking operations for web APIs
- **Memory Management**: Automatic model unloading based on usage patterns

### Scaling Guidelines

- **Horizontal Scaling**: Deploy multiple instances behind a load balancer
- **Vertical Scaling**: Increase CPU/memory for better single-instance performance
- **Model Optimization**: Use quantized models for faster inference
- **Caching Strategy**: Implement Redis for cross-instance model caching

## 📈 Monitoring

### Metrics Dashboard

The pipeline includes built-in monitoring with the following metrics:

- **Request Metrics**: Response time, throughput, error rates
- **Model Metrics**: Prediction accuracy, model load times
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Sentiment distribution, entity frequencies

### Health Checks

```python
from src.monitoring import HealthChecker

health_checker = HealthChecker()
health_status = health_checker.get_health_status()

print(f"Overall Status: {health_status['status']}")
for component, status in health_status['components'].items():
    print(f"{component}: {status['status']}")
```

### Alerts

Configure alerts in `config/alerts.yaml`:

```yaml
alerts:
  - name: "High Response Time"
    condition: "avg_response_time > 200"
    severity: "warning"
    
  - name: "High Error Rate"
    condition: "error_rate > 5"
    severity: "critical"
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_ingestion.py
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and stress testing
- **Model Tests**: ML model accuracy and performance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system architecture
- [API Documentation](docs/API.md) - Complete API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Performance Guide](docs/PERFORMANCE.md) - Optimization and tuning guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the model implementations
- [Apache Spark](https://spark.apache.org/) for distributed processing capabilities
- [Apache Kafka](https://kafka.apache.org/) for streaming data infrastructure
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For questions and support:

- 📧 Email: fenil@fenilsonani.com
- 🐛 Issues: [GitHub Issues](https://github.com/fenilsonani/nlppipeline/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/fenilsonani/nlppipeline/discussions)

---

**Built with ❤️ for enterprise NLP applications**