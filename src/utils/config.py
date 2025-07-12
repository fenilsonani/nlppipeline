"""Configuration management for NLP Pipeline."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic: str = os.getenv("KAFKA_TOPIC", "nlp_pipeline_input")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "nlp_pipeline_consumer")
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500


@dataclass
class SparkConfig:
    """Spark configuration."""
    master: str = os.getenv("SPARK_MASTER", "local[*]")
    app_name: str = os.getenv("SPARK_APP_NAME", "NLP_Pipeline")
    executor_memory: str = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    driver_memory: str = os.getenv("SPARK_DRIVER_MEMORY", "2g")
    max_cores: int = int(os.getenv("SPARK_MAX_CORES", "4"))


@dataclass
class ModelConfig:
    """Model configuration."""
    bert_model_path: str = os.getenv("BERT_MODEL_PATH", "bert-base-uncased")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./data/models")
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/nlp_pipeline")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class APIConfig:
    """API configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8001"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_metrics: bool = True
    enable_tracing: bool = False


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    timeout: int = 300  # seconds
    retry_count: int = 3
    retry_delay: int = 5  # seconds


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_path: Path = Path(os.getenv("DATA_PATH", "./data"))
    output_path: Path = Path(os.getenv("OUTPUT_PATH", "./output"))
    log_path: Path = Path("./logs")
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.kafka = KafkaConfig()
        self.spark = SparkConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        self.processing = ProcessingConfig()
        self.storage = StorageConfig()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configurations
        for section, values in config_data.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'kafka': self.kafka.__dict__,
            'spark': self.spark.__dict__,
            'model': self.model.__dict__,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__,
            'processing': self.processing.__dict__,
            'storage': {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.storage.__dict__.items()}
        }
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML file."""
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Global configuration instance
config = Config()