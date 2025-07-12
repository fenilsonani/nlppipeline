"""
Kafka consumer for streaming data ingestion.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from confluent_kafka import Consumer, KafkaError, KafkaException
import time

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Configuration for Kafka consumer."""
    bootstrap_servers: str
    group_id: str
    topic: str
    auto_offset_reset: str = 'latest'
    enable_auto_commit: bool = True
    session_timeout_ms: int = 30000
    max_poll_interval_ms: int = 300000
    
    @classmethod
    def from_env(cls) -> 'KafkaConfig':
        """Load configuration from environment variables."""
        return cls(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            group_id=os.getenv('KAFKA_GROUP_ID', 'nlp-pipeline-consumer'),
            topic=os.getenv('KAFKA_TOPIC', 'nlp-input'),
            auto_offset_reset=os.getenv('KAFKA_AUTO_OFFSET_RESET', 'latest'),
            enable_auto_commit=os.getenv('KAFKA_ENABLE_AUTO_COMMIT', 'true').lower() == 'true',
            session_timeout_ms=int(os.getenv('KAFKA_SESSION_TIMEOUT_MS', '30000')),
            max_poll_interval_ms=int(os.getenv('KAFKA_MAX_POLL_INTERVAL_MS', '300000'))
        )


class KafkaStreamConsumer:
    """Kafka consumer for streaming data ingestion with error handling and retries."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """
        Initialize Kafka consumer.
        
        Args:
            config: Kafka configuration. If None, loads from environment.
        """
        self.config = config or KafkaConfig.from_env()
        self.consumer = None
        self.running = False
        self._init_consumer()
    
    def _init_consumer(self):
        """Initialize Kafka consumer with configuration."""
        try:
            conf = {
                'bootstrap.servers': self.config.bootstrap_servers,
                'group.id': self.config.group_id,
                'auto.offset.reset': self.config.auto_offset_reset,
                'enable.auto.commit': self.config.enable_auto_commit,
                'session.timeout.ms': self.config.session_timeout_ms,
                'max.poll.interval.ms': self.config.max_poll_interval_ms,
                'error_cb': self._error_callback
            }
            
            self.consumer = Consumer(conf)
            self.consumer.subscribe([self.config.topic])
            logger.info(f"Kafka consumer initialized for topic: {self.config.topic}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def _error_callback(self, err):
        """Handle Kafka errors."""
        logger.error(f"Kafka error: {err}")
    
    def consume_messages(self, 
                        process_func: Callable[[Dict[str, Any]], None],
                        batch_size: int = 100,
                        poll_timeout: float = 1.0) -> None:
        """
        Consume messages from Kafka and process them.
        
        Args:
            process_func: Function to process each message
            batch_size: Number of messages to process in batch
            poll_timeout: Timeout for polling messages
        """
        self.running = True
        messages_batch = []
        
        try:
            while self.running:
                msg = self.consumer.poll(timeout=poll_timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info(f"Reached end of partition {msg.topic()}/{msg.partition()}")
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    # Decode message
                    value = json.loads(msg.value().decode('utf-8'))
                    messages_batch.append(value)
                    
                    # Process batch if full
                    if len(messages_batch) >= batch_size:
                        self._process_batch(messages_batch, process_func)
                        messages_batch = []
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._handle_processing_error(msg, e)
            
            # Process remaining messages
            if messages_batch:
                self._process_batch(messages_batch, process_func)
                
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except KafkaException as e:
            logger.error(f"Kafka exception: {e}")
            raise
        finally:
            self.close()
    
    def _process_batch(self, batch: list, process_func: Callable) -> None:
        """Process a batch of messages with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                for message in batch:
                    process_func(message)
                self.consumer.commit()
                return
            except Exception as e:
                logger.error(f"Batch processing failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error("Max retries exceeded, skipping batch")
                    # Could implement dead letter queue here
    
    def _handle_processing_error(self, msg, error: Exception) -> None:
        """Handle processing errors with retry logic."""
        logger.error(f"Processing error for message at offset {msg.offset()}: {error}")
        # Could implement dead letter queue or error topic here
    
    def stop(self):
        """Stop consuming messages."""
        self.running = False
        logger.info("Stopping Kafka consumer...")
    
    def close(self):
        """Close Kafka consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        if not self.consumer:
            return {}
        
        try:
            # Get consumer position
            partitions = self.consumer.assignment()
            positions = {}
            for partition in partitions:
                position = self.consumer.position([partition])
                positions[f"{partition.topic}-{partition.partition}"] = position[0].offset
            
            return {
                "assigned_partitions": len(partitions),
                "positions": positions
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def process_message(message: Dict[str, Any]):
        """Example message processor."""
        print(f"Processing message: {message}")
    
    consumer = KafkaStreamConsumer()
    try:
        consumer.consume_messages(process_message)
    except KeyboardInterrupt:
        consumer.stop()