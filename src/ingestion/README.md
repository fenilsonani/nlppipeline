# Data Ingestion Module

This module provides comprehensive data ingestion capabilities for the NLP Pipeline, supporting both streaming and batch data processing.

## Features

- **Kafka Streaming**: Real-time data ingestion from Kafka topics
- **Spark Processing**: Scalable stream processing with Apache Spark
- **Batch Loading**: Support for multiple file formats (CSV, JSON, Parquet, YAML, TXT)
- **Error Handling**: Configurable error handling with retry mechanisms
- **Environment Configuration**: All components configurable via environment variables

## Installation

```bash
pip install -r requirements.txt
```

## Components

### 1. Kafka Consumer (`kafka_consumer.py`)

Consumes streaming data from Kafka topics with automatic retries and error handling.

**Usage:**
```python
from src.ingestion import KafkaStreamConsumer

def process_message(message):
    print(f"Processing: {message}")

consumer = KafkaStreamConsumer()
consumer.consume_messages(process_message, batch_size=100)
```

**Environment Variables:**
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses (default: localhost:9092)
- `KAFKA_GROUP_ID`: Consumer group ID (default: nlp-pipeline-consumer)
- `KAFKA_TOPIC`: Topic to consume from (default: nlp-input)
- `KAFKA_AUTO_OFFSET_RESET`: Where to start reading (default: latest)

### 2. Spark Processor (`spark_processor.py`)

Processes streaming data using Apache Spark with support for various output sinks.

**Usage:**
```python
from src.ingestion import SparkStreamProcessor
from pyspark.sql.functions import col

def process_data(df):
    return df.withColumn("text_length", col("text").length())

processor = SparkStreamProcessor()
stream = processor.create_kafka_stream()
query = processor.process_stream(stream, process_data, output_sink="console")
processor.await_termination()
```

**Environment Variables:**
- `SPARK_APP_NAME`: Spark application name (default: NLP-Pipeline-Streaming)
- `SPARK_MASTER`: Spark master URL (default: local[*])
- `SPARK_CHECKPOINT_DIR`: Checkpoint directory (default: /tmp/spark-checkpoints)
- `SPARK_TRIGGER_INTERVAL`: Processing trigger interval (default: 10 seconds)

### 3. Data Loader (`data_loader.py`)

Loads batch data from various file formats with parallel processing support.

**Usage:**
```python
from src.ingestion import DataLoader

loader = DataLoader()

# Load single file
for record in loader.load_file("data.csv"):
    print(record)

# Load directory
for record in loader.load_directory("data/", pattern="*.json"):
    print(record)

# Batch load with processing
def process(record):
    record['processed'] = True
    return record

results = loader.load_batch(["file1.csv", "data_dir/"], process_func=process)
```

**Environment Variables:**
- `LOADER_BATCH_SIZE`: Records per batch (default: 1000)
- `LOADER_MAX_WORKERS`: Parallel workers (default: 4)
- `LOADER_FORMATS`: Supported formats (default: csv,json,txt,parquet,yaml)
- `LOADER_ERROR_HANDLING`: Error strategy - skip/raise/log (default: log)

## Example Integration

```python
from src.ingestion import KafkaStreamConsumer, SparkStreamProcessor, DataLoader

# 1. Load historical data
loader = DataLoader()
historical_data = loader.load_batch(["historical_data/"])

# 2. Set up streaming pipeline
consumer = KafkaStreamConsumer()
processor = SparkStreamProcessor()

# 3. Process streaming data
stream = processor.create_kafka_stream()
processed_stream = processor.process_stream(
    stream,
    lambda df: df.filter(df.text.isNotNull()),
    output_sink="parquet",
    output_options={"path": "output/processed_data"}
)

# 4. Monitor progress
metrics = processor.monitor_stream()
print(f"Processing rate: {metrics['processedRowsPerSecond']} records/sec")
```

## Error Handling

All components include comprehensive error handling:

- **Retries**: Automatic retry with exponential backoff
- **Dead Letter Queue**: Failed messages can be sent to error topics
- **Checkpointing**: Save and resume processing state
- **Monitoring**: Built-in metrics for tracking performance

## Performance Optimization

- Batch processing for improved throughput
- Parallel file loading with thread pools
- Spark adaptive query execution
- Configurable batch sizes and intervals
- Memory-efficient streaming with generators

## Testing

Run the example scripts in each module:

```bash
python -m src.ingestion.kafka_consumer
python -m src.ingestion.spark_processor
python -m src.ingestion.data_loader
```