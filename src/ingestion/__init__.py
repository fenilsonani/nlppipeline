"""
Data ingestion module for NLP Pipeline.

This module provides components for ingesting data from various sources:
- Kafka streaming ingestion
- Spark streaming processing
- Batch file loading

Components:
- KafkaStreamConsumer: Consume streaming data from Kafka topics
- SparkStreamProcessor: Process streaming data with Apache Spark
- DataLoader: Load batch data from various file formats
"""

# Make Kafka and Spark optional
try:
    from .kafka_consumer import KafkaStreamConsumer, KafkaConfig
    KafkaConsumer = KafkaStreamConsumer  # Alias for backward compatibility
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaStreamConsumer = None
    KafkaConsumer = None
    KafkaConfig = None
    KAFKA_AVAILABLE = False

try:
    from .spark_processor import SparkStreamProcessor, SparkConfig
    SparkProcessor = SparkStreamProcessor  # Alias for backward compatibility
    SPARK_AVAILABLE = True
except ImportError:
    SparkStreamProcessor = None
    SparkProcessor = None
    SparkConfig = None
    SPARK_AVAILABLE = False

# DataLoader should always be available
from .data_loader import DataLoader, LoaderConfig

__all__ = [
    'KafkaStreamConsumer',
    'KafkaConsumer',
    'KafkaConfig',
    'SparkStreamProcessor',
    'SparkProcessor', 
    'SparkConfig',
    'DataLoader',
    'LoaderConfig',
    'KAFKA_AVAILABLE',
    'SPARK_AVAILABLE'
]

# Version info
__version__ = '0.1.0'