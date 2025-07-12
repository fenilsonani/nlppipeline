"""Logging configuration for NLP Pipeline."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from pythonjsonlogger import jsonlogger

from .config import config


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


def setup_logger(
    name: str = "nlp_pipeline",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Log level (defaults to config.monitoring.log_level)
        log_file: Log file path (optional)
        json_format: Use JSON format for logs
    
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.monitoring.log_level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if json_format:
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to log messages."""
    
    def process(self, msg, kwargs):
        """Add context to log messages."""
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """
    Get logger with context.
    
    Args:
        name: Logger name
        **context: Additional context to add to all log messages
    
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    
    return LoggerAdapter(logger, context)


# Module-specific loggers
ingestion_logger = get_logger("nlp_pipeline.ingestion")
preprocessing_logger = get_logger("nlp_pipeline.preprocessing")
model_logger = get_logger("nlp_pipeline.models")
postprocessing_logger = get_logger("nlp_pipeline.postprocessing")
monitoring_logger = get_logger("nlp_pipeline.monitoring")
api_logger = get_logger("nlp_pipeline.api")


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log operation timing."""
        self.logger.info(
            f"Performance metric: {operation}",
            extra={
                "metric_type": "timing",
                "operation": operation,
                "duration_ms": duration * 1000,
                **kwargs
            }
        )
    
    def log_throughput(self, operation: str, count: int, duration: float, **kwargs):
        """Log operation throughput."""
        throughput = count / duration if duration > 0 else 0
        self.logger.info(
            f"Throughput metric: {operation}",
            extra={
                "metric_type": "throughput",
                "operation": operation,
                "count": count,
                "duration_s": duration,
                "throughput_per_s": throughput,
                **kwargs
            }
        )
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float, **kwargs):
        """Log resource usage."""
        self.logger.info(
            "Resource usage metric",
            extra={
                "metric_type": "resource_usage",
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                **kwargs
            }
        )


# Create performance logger
perf_logger = PerformanceLogger(get_logger("nlp_pipeline.performance"))