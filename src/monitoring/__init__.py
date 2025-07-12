"""
Monitoring module for the NLP pipeline.

This module provides comprehensive monitoring capabilities including:
- Performance metrics collection
- System health monitoring  
- Alert management and notifications
- Prometheus metrics export
- HTTP dashboard endpoints

Main components:
- MetricsCollector: Collects and stores performance metrics
- HealthChecker: Monitors system and service health
- AlertManager: Handles alerts and notifications
- MonitoringDashboard: Provides HTTP endpoints and Prometheus metrics
"""

from .metrics_collector import MetricsCollector, metrics_collector
from .health_checker import HealthChecker, HealthStatus, HealthCheck, health_checker
from .alert_manager import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus,
    NotificationChannel, EmailNotificationChannel, WebhookNotificationChannel, 
    SlackNotificationChannel, alert_manager, setup_default_alert_rules
)
from .dashboard import (
    MonitoringDashboard, PrometheusMetrics, monitoring_dashboard, 
    start_monitoring_server
)

__all__ = [
    # Core classes
    'MetricsCollector',
    'HealthChecker', 
    'AlertManager',
    'MonitoringDashboard',
    
    # Health monitoring
    'HealthStatus',
    'HealthCheck',
    
    # Alert management
    'Alert',
    'AlertRule', 
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'EmailNotificationChannel',
    'WebhookNotificationChannel',
    'SlackNotificationChannel',
    
    # Prometheus metrics
    'PrometheusMetrics',
    
    # Global instances
    'metrics_collector',
    'health_checker',
    'alert_manager', 
    'monitoring_dashboard',
    
    # Utility functions
    'setup_default_alert_rules',
    'start_monitoring_server'
]

# Version info
__version__ = "1.0.0"