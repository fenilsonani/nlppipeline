# NLP Pipeline Monitoring Module

A comprehensive monitoring solution for the NLP pipeline that provides metrics collection, health monitoring, alerting, and Prometheus integration.

## Overview

The monitoring module consists of five main components:

1. **MetricsCollector** - Collects and stores performance metrics
2. **HealthChecker** - System and service health monitoring
3. **AlertManager** - Alert management and notifications
4. **Dashboard** - HTTP endpoints and Prometheus metrics export
5. **Integration** - Seamless integration between all components

## Features

### Metrics Collection
- Document processing throughput
- Model inference latency  
- System resource usage (CPU, memory, disk)
- Error tracking and counting
- Custom metrics with labels
- Histogram statistics (percentiles, averages)
- Thread-safe operations

### Health Monitoring
- Configurable health checks with intervals
- Critical vs non-critical check classification
- Automatic status aggregation
- Async health check execution
- Timeout handling

### Alert Management
- Rule-based alerting with custom conditions
- Multiple severity levels (LOW, MEDIUM, HIGH, CRITICAL)
- Cooldown periods to prevent alert spam
- Multiple notification channels:
  - Email (SMTP)
  - Slack webhooks
  - Generic webhooks
- Alert history and statistics

### Dashboard & Metrics Export
- FastAPI-based HTTP dashboard
- Prometheus metrics endpoint (`/metrics`)
- JSON metrics endpoint (`/metrics/json`)
- Health check endpoint (`/health`)
- Alert management endpoints
- Real-time metrics updates

## Quick Start

### Basic Usage

```python
import asyncio
from src.monitoring import (
    metrics_collector, health_checker, alert_manager,
    monitoring_dashboard, setup_default_alert_rules
)

async def main():
    # Setup default alert rules
    setup_default_alert_rules()
    
    # Start health monitoring
    await health_checker.start()
    
    # Collect some metrics
    with metrics_collector.timer('document_processing'):
        # Your processing code here
        pass
    
    metrics_collector.increment_counter('documents_processed')
    metrics_collector.set_gauge('queue_size', 42)
    
    # Start the monitoring dashboard
    await monitoring_dashboard.start()  # Available at http://localhost:8080

asyncio.run(main())
```

### Custom Health Checks

```python
def check_database():
    # Your database connectivity check
    return True  # or False

health_checker.register_check(
    "database",
    check_database,
    interval=30,     # Check every 30 seconds
    critical=True    # Critical for overall health
)
```

### Custom Alert Rules

```python
from src.monitoring import AlertRule, AlertSeverity

# Alert when processing queue gets too large
queue_alert = AlertRule(
    name="high_queue_size",
    condition=lambda x: x > 100,
    severity=AlertSeverity.HIGH,
    message_template="Queue size is {value}, exceeding threshold of 100",
    source="queue_size",
    threshold=100.0,
    cooldown_period=300  # 5 minutes
)

alert_manager.add_rule(queue_alert)
```

### Notification Channels

```python
from src.monitoring import SlackNotificationChannel, EmailNotificationChannel

# Slack notifications
slack_channel = SlackNotificationChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#alerts"
)
alert_manager.add_notification_channel(slack_channel)

# Email notifications  
email_channel = EmailNotificationChannel(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@company.com",
    password="app-password",
    from_email="alerts@company.com",
    to_emails=["admin@company.com"]
)
alert_manager.add_notification_channel(email_channel)
```

## API Endpoints

When the dashboard is running, the following endpoints are available:

- `GET /` - API overview and available endpoints
- `GET /health` - Health check status (returns 200/503 based on health)
- `GET /metrics` - Prometheus metrics format
- `GET /metrics/json` - JSON formatted metrics
- `GET /alerts` - Active alerts
- `GET /alerts/history?limit=100` - Alert history
- `GET /stats` - System statistics summary

## Prometheus Integration

The module automatically exports metrics in Prometheus format at `/metrics`:

```
# Document processing metrics
nlp_documents_processed_total{status="success",model_type="bert"} 150
nlp_document_processing_duration_seconds_bucket{model_type="bert",le="1.0"} 45

# System metrics  
system_cpu_percent 23.5
system_memory_percent 67.2
system_disk_percent 45.1

# Model inference metrics
nlp_model_inference_duration_seconds{model_name="sentiment",model_type="transformer"} 0.145

# Pipeline health
nlp_pipeline_health{component="database"} 1
nlp_pipeline_health{component="model_service"} 1

# Active alerts
nlp_active_alerts_total{severity="high"} 2
```

## Default Alert Rules

The module includes pre-configured alert rules:

- **high_cpu_usage**: CPU > 80%
- **high_memory_usage**: Memory > 85%  
- **high_disk_usage**: Disk > 90%
- **slow_document_processing**: Processing time > 5 seconds
- **high_model_latency**: Model inference > 2 seconds
- **low_processing_throughput**: < 10 documents per minute

## Configuration

### Environment Variables

```bash
# Dashboard configuration
MONITORING_HOST=0.0.0.0
MONITORING_PORT=8080

# Alert thresholds
CPU_ALERT_THRESHOLD=80
MEMORY_ALERT_THRESHOLD=85
DISK_ALERT_THRESHOLD=90

# SMTP configuration (for email alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@company.com
SMTP_PASSWORD=app-password
```

## Performance Considerations

- Metrics are stored in memory with configurable history limits
- System metrics collection runs in background thread
- Health checks run asynchronously with configurable intervals
- Alert evaluation has built-in cooldown periods
- Prometheus metrics are generated on-demand

## Dependencies

Required packages:
- `fastapi>=0.68.0` - Web framework for dashboard
- `uvicorn>=0.15.0` - ASGI server
- `prometheus-client>=0.11.0` - Prometheus metrics
- `aiohttp>=3.8.0` - HTTP client for webhooks
- `psutil>=5.8.0` - System metrics

Optional (for email alerts):
- Python standard library `email` and `smtplib` modules

## Examples

See the `examples/` directory for complete usage examples:

- `examples/monitoring_usage.py` - Comprehensive usage example
- `examples/simple_monitoring_test.py` - Basic functionality test

## Error Handling

The module includes comprehensive error handling:

- Failed health checks are logged and counted
- Alert notification failures are logged but don't stop processing
- Metrics collection errors are isolated and logged
- Dashboard endpoints return appropriate HTTP status codes
- Graceful degradation when optional dependencies are missing

## Thread Safety

All components are designed to be thread-safe:

- MetricsCollector uses locks for concurrent access
- HealthChecker uses asyncio for concurrent health checks
- AlertManager handles concurrent metric evaluation
- Dashboard serves multiple concurrent requests

## Monitoring the Monitor

The monitoring module itself can be monitored:

- Health checks include self-monitoring
- Metrics about metrics collection (meta-metrics)
- Alert statistics and performance
- Dashboard availability and response times