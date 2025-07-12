"""
Example pipeline demonstrating the usage of all ingestion components.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any

from kafka_consumer import KafkaStreamConsumer
from spark_processor import SparkStreamProcessor
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Example ingestion pipeline combining all components."""
    
    def __init__(self):
        self.kafka_consumer = None
        self.spark_processor = None
        self.data_loader = DataLoader()
        self.stats = {
            'batch_records': 0,
            'stream_records': 0,
            'errors': 0
        }
    
    def load_historical_data(self, data_path: str) -> None:
        """Load historical data from files."""
        logger.info(f"Loading historical data from: {data_path}")
        
        def preprocess_batch(record: Dict[str, Any]) -> Dict[str, Any]:
            """Preprocess batch records."""
            record['source'] = 'batch'
            record['processed_at'] = datetime.utcnow().isoformat()
            return record
        
        try:
            results = self.data_loader.load_batch(
                sources=[data_path],
                process_func=preprocess_batch
            )
            self.stats['batch_records'] = len(results)
            logger.info(f"Loaded {len(results)} historical records")
            
            # Here you would typically save to a database or data lake
            # For demo, just log some stats
            if results:
                logger.info(f"Sample record: {results[0]}")
                
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            self.stats['errors'] += 1
    
    def setup_streaming_pipeline(self) -> None:
        """Set up Kafka to Spark streaming pipeline."""
        logger.info("Setting up streaming pipeline")
        
        try:
            # Initialize components
            self.kafka_consumer = KafkaStreamConsumer()
            self.spark_processor = SparkStreamProcessor()
            
            # Create Kafka stream in Spark
            stream = self.spark_processor.create_kafka_stream()
            
            # Define processing logic
            from pyspark.sql.functions import col, current_timestamp, length
            
            def process_stream_data(df):
                """Process streaming data."""
                return (df
                       .withColumn("processed_timestamp", current_timestamp())
                       .withColumn("text_length", length(col("text")))
                       .filter(col("text").isNotNull())
                       .filter(col("text_length") > 0))
            
            # Start processing
            query = self.spark_processor.process_stream(
                stream,
                process_stream_data,
                output_sink="console",  # Change to "kafka" or "parquet" for production
                output_options={
                    "truncate": "false",
                    "numRows": "20"
                }
            )
            
            logger.info("Streaming pipeline started successfully")
            return query
            
        except Exception as e:
            logger.error(f"Failed to setup streaming pipeline: {e}")
            self.stats['errors'] += 1
            raise
    
    def run_hybrid_pipeline(self, 
                           batch_data_path: str,
                           streaming_duration: int = 60) -> None:
        """
        Run hybrid batch + streaming pipeline.
        
        Args:
            batch_data_path: Path to batch data
            streaming_duration: How long to run streaming (seconds)
        """
        logger.info("Starting hybrid ingestion pipeline")
        
        # Step 1: Load historical/batch data
        if os.path.exists(batch_data_path):
            self.load_historical_data(batch_data_path)
        else:
            logger.warning(f"Batch data path not found: {batch_data_path}")
        
        # Step 2: Start streaming pipeline
        try:
            query = self.setup_streaming_pipeline()
            
            # Step 3: Monitor streaming for specified duration
            import time
            start_time = time.time()
            
            while time.time() - start_time < streaming_duration:
                if query and query.isActive:
                    metrics = self.spark_processor.monitor_stream(query)
                    logger.info(f"Stream metrics: {metrics}")
                    time.sleep(10)  # Check every 10 seconds
                else:
                    logger.warning("Streaming query is not active")
                    break
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up pipeline resources")
        
        if self.spark_processor:
            self.spark_processor.stop()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        # Log final stats
        logger.info(f"Pipeline statistics: {self.stats}")


def main():
    """Main entry point."""
    # Example configuration via environment variables
    os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'localhost:9092'
    os.environ['KAFKA_TOPIC'] = 'nlp-input'
    os.environ['SPARK_APP_NAME'] = 'NLP-Ingestion-Example'
    os.environ['LOADER_BATCH_SIZE'] = '500'
    
    # Create and run pipeline
    pipeline = IngestionPipeline()
    
    try:
        # Run hybrid pipeline
        # - Load batch data from 'data/' directory
        # - Stream for 60 seconds
        pipeline.run_hybrid_pipeline(
            batch_data_path="data/",
            streaming_duration=60
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())