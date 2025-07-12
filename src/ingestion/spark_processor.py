"""
Spark streaming processor for real-time data processing.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.streaming import StreamingQuery

logger = logging.getLogger(__name__)


@dataclass
class SparkConfig:
    """Configuration for Spark streaming."""
    app_name: str
    master: str
    kafka_bootstrap_servers: str
    kafka_topic: str
    checkpoint_location: str
    output_mode: str = "append"
    trigger_interval: str = "10 seconds"
    max_offsets_per_trigger: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> 'SparkConfig':
        """Load configuration from environment variables."""
        return cls(
            app_name=os.getenv('SPARK_APP_NAME', 'NLP-Pipeline-Streaming'),
            master=os.getenv('SPARK_MASTER', 'local[*]'),
            kafka_bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            kafka_topic=os.getenv('KAFKA_TOPIC', 'nlp-input'),
            checkpoint_location=os.getenv('SPARK_CHECKPOINT_DIR', '/tmp/spark-checkpoints'),
            output_mode=os.getenv('SPARK_OUTPUT_MODE', 'append'),
            trigger_interval=os.getenv('SPARK_TRIGGER_INTERVAL', '10 seconds'),
            max_offsets_per_trigger=int(os.getenv('SPARK_MAX_OFFSETS_PER_TRIGGER', '10000')) 
                if os.getenv('SPARK_MAX_OFFSETS_PER_TRIGGER') else None
        )


class SparkStreamProcessor:
    """Spark streaming processor with error handling and monitoring."""
    
    def __init__(self, config: Optional[SparkConfig] = None):
        """
        Initialize Spark streaming processor.
        
        Args:
            config: Spark configuration. If None, loads from environment.
        """
        self.config = config or SparkConfig.from_env()
        self.spark = None
        self.streaming_query = None
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session with configuration."""
        try:
            self.spark = (SparkSession.builder
                         .appName(self.config.app_name)
                         .master(self.config.master)
                         .config("spark.sql.streaming.checkpointLocation", self.config.checkpoint_location)
                         .config("spark.sql.adaptive.enabled", "true")
                         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                         .getOrCreate())
            
            # Set log level
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark session initialized: {self.config.app_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
            raise
    
    def create_kafka_stream(self, 
                           schema: Optional[StructType] = None,
                           starting_offsets: str = "latest") -> DataFrame:
        """
        Create Kafka stream source.
        
        Args:
            schema: Schema for the JSON data. If None, uses default schema.
            starting_offsets: Starting position for reading data.
            
        Returns:
            DataFrame with streaming data.
        """
        if schema is None:
            schema = self._get_default_schema()
        
        try:
            # Read from Kafka
            kafka_options = {
                "kafka.bootstrap.servers": self.config.kafka_bootstrap_servers,
                "subscribe": self.config.kafka_topic,
                "startingOffsets": starting_offsets,
                "failOnDataLoss": "false"
            }
            
            if self.config.max_offsets_per_trigger:
                kafka_options["maxOffsetsPerTrigger"] = str(self.config.max_offsets_per_trigger)
            
            df = (self.spark
                  .readStream
                  .format("kafka")
                  .options(**kafka_options)
                  .load())
            
            # Parse JSON data
            parsed_df = (df
                        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp")
                        .select(col("key"),
                               from_json(col("value"), schema).alias("data"),
                               col("timestamp")))
            
            # Flatten the structure
            result_df = parsed_df.select("key", "data.*", "timestamp")
            
            logger.info("Kafka stream created successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to create Kafka stream: {e}")
            raise
    
    def _get_default_schema(self) -> StructType:
        """Get default schema for JSON data."""
        return StructType([
            StructField("id", StringType(), True),
            StructField("text", StringType(), True),
            StructField("metadata", StringType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
    
    def process_stream(self,
                      input_stream: DataFrame,
                      process_func: Callable[[DataFrame], DataFrame],
                      output_sink: str = "console",
                      output_options: Optional[Dict[str, Any]] = None) -> StreamingQuery:
        """
        Process streaming data with custom function.
        
        Args:
            input_stream: Input streaming DataFrame
            process_func: Function to process the DataFrame
            output_sink: Output sink type (console, kafka, file, etc.)
            output_options: Options for the output sink
            
        Returns:
            StreamingQuery object
        """
        try:
            # Apply processing function
            processed_stream = process_func(input_stream)
            
            # Configure output
            writer = (processed_stream
                     .writeStream
                     .outputMode(self.config.output_mode)
                     .trigger(processingTime=self.config.trigger_interval))
            
            # Add output options
            if output_options:
                for key, value in output_options.items():
                    writer = writer.option(key, value)
            
            # Start query based on sink type
            if output_sink == "console":
                query = writer.format("console").start()
            elif output_sink == "kafka":
                # Convert to Kafka format
                kafka_stream = processed_stream.select(
                    col("id").alias("key"),
                    to_json(struct("*")).alias("value")
                )
                query = (kafka_stream
                        .writeStream
                        .format("kafka")
                        .option("kafka.bootstrap.servers", self.config.kafka_bootstrap_servers)
                        .option("topic", output_options.get("topic", "nlp-output"))
                        .start())
            elif output_sink == "parquet":
                query = (writer
                        .format("parquet")
                        .option("path", output_options.get("path", "/tmp/spark-output"))
                        .start())
            else:
                raise ValueError(f"Unsupported output sink: {output_sink}")
            
            self.streaming_query = query
            logger.info(f"Streaming query started with sink: {output_sink}")
            return query
            
        except Exception as e:
            logger.error(f"Failed to process stream: {e}")
            raise
    
    def monitor_stream(self, query: Optional[StreamingQuery] = None) -> Dict[str, Any]:
        """
        Monitor streaming query progress.
        
        Args:
            query: StreamingQuery to monitor. If None, uses current query.
            
        Returns:
            Dictionary with monitoring metrics.
        """
        query = query or self.streaming_query
        if not query:
            return {"error": "No active streaming query"}
        
        try:
            progress = query.lastProgress
            if progress:
                return {
                    "id": query.id,
                    "name": query.name,
                    "isActive": query.isActive,
                    "inputRowsPerSecond": progress.get("inputRowsPerSecond", 0),
                    "processedRowsPerSecond": progress.get("processedRowsPerSecond", 0),
                    "batchId": progress.get("batchId", 0),
                    "durationMs": progress.get("durationMs", {}),
                    "sources": progress.get("sources", [])
                }
            return {"id": query.id, "isActive": query.isActive}
            
        except Exception as e:
            logger.error(f"Failed to get stream metrics: {e}")
            return {"error": str(e)}
    
    def stop_stream(self, query: Optional[StreamingQuery] = None):
        """Stop streaming query."""
        query = query or self.streaming_query
        if query and query.isActive:
            query.stop()
            logger.info("Streaming query stopped")
    
    def stop(self):
        """Stop Spark session and all queries."""
        if self.streaming_query:
            self.stop_stream()
        
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def await_termination(self, timeout: Optional[int] = None):
        """
        Wait for streaming query termination.
        
        Args:
            timeout: Timeout in seconds. If None, waits indefinitely.
        """
        if self.streaming_query:
            try:
                if timeout:
                    self.streaming_query.awaitTermination(timeout)
                else:
                    self.streaming_query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Streaming interrupted by user")
                self.stop()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def process_data(df: DataFrame) -> DataFrame:
        """Example processing function."""
        # Add text length column
        return df.withColumn("text_length", col("text").length())
    
    processor = SparkStreamProcessor()
    
    try:
        # Create stream
        stream = processor.create_kafka_stream()
        
        # Process and output to console
        query = processor.process_stream(
            stream, 
            process_data, 
            output_sink="console"
        )
        
        # Monitor progress
        import time
        for _ in range(5):
            time.sleep(10)
            metrics = processor.monitor_stream()
            print(f"Stream metrics: {metrics}")
        
        # Wait for termination
        processor.await_termination()
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
    finally:
        processor.stop()