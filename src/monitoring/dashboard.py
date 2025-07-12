"""
Prometheus metrics dashboard and HTTP endpoints for monitoring.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
import uvicorn
import logging
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, 
    CollectorRegistry, CONTENT_TYPE_LATEST
)
import json

from .metrics_collector import metrics_collector
from .health_checker import health_checker
from .alert_manager import alert_manager


class PrometheusMetrics:
    """Prometheus metrics exporter."""
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Document processing metrics
        self.documents_processed = Counter(
            'nlp_documents_processed_total',
            'Total number of documents processed',
            ['status', 'model_type'],
            registry=self.registry
        )
        
        self.document_processing_duration = Histogram(
            'nlp_document_processing_duration_seconds',
            'Time spent processing documents',
            ['model_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Model inference metrics
        self.model_inference_duration = Histogram(
            'nlp_model_inference_duration_seconds',
            'Time spent on model inference',
            ['model_name', 'model_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.model_predictions = Counter(
            'nlp_model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'model_type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_percent = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_used_bytes = Gauge(
            'system_memory_used_bytes',
            'System memory used in bytes',
            registry=self.registry
        )
        
        self.system_memory_percent = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_used_bytes = Gauge(
            'system_disk_used_bytes',
            'System disk used in bytes',
            registry=self.registry
        )
        
        self.system_disk_percent = Gauge(
            'system_disk_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Error tracking
        self.processing_errors = Counter(
            'nlp_processing_errors_total',
            'Total number of processing errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Throughput metrics
        self.documents_per_minute = Gauge(
            'nlp_documents_per_minute',
            'Documents processed per minute',
            registry=self.registry
        )
        
        # Pipeline health
        self.pipeline_health = Gauge(
            'nlp_pipeline_health',
            'Pipeline health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        # Active alerts
        self.active_alerts = Gauge(
            'nlp_active_alerts_total',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )
    
    def update_from_metrics_collector(self):
        """Update Prometheus metrics from the metrics collector."""
        try:
            # Get all metrics from collector
            metrics = metrics_collector.get_metrics()
            
            # Update gauges from collected metrics
            gauges = metrics.get('gauges', {})
            for metric_name, value in gauges.items():
                if 'system_cpu_percent' in metric_name:
                    self.system_cpu_percent.set(value)
                elif 'system_memory_used_bytes' in metric_name:
                    self.system_memory_used_bytes.set(value)
                elif 'system_memory_percent' in metric_name:
                    self.system_memory_percent.set(value)
                elif 'system_disk_used_bytes' in metric_name:
                    self.system_disk_used_bytes.set(value)
                elif 'system_disk_percent' in metric_name:
                    self.system_disk_percent.set(value)
            
            # Update histograms - we'll use the latest values for demonstration
            histograms = metrics.get('histograms', {})
            for metric_name, stats in histograms.items():
                if 'document_processing_duration' in metric_name:
                    # For Prometheus, we need to observe individual values
                    # This is a simplified approach - in practice, you'd observe each value
                    if stats.get('avg'):
                        # Extract labels from metric name if present
                        labels = self._extract_labels(metric_name)
                        self.document_processing_duration.labels(**labels).observe(stats['avg'])
                
                elif 'model_inference_duration' in metric_name:
                    if stats.get('avg'):
                        labels = self._extract_labels(metric_name)
                        self.model_inference_duration.labels(**labels).observe(stats['avg'])
            
            # Calculate and update throughput
            doc_count = metrics.get('counters', {}).get('documents_processed', 0)
            # Simple calculation - in practice, you'd track this over time windows
            if hasattr(self, '_last_doc_count'):
                throughput = max(0, doc_count - self._last_doc_count)
                self.documents_per_minute.set(throughput)
            self._last_doc_count = doc_count
            
        except Exception as e:
            logging.error(f"Error updating Prometheus metrics: {e}")
    
    def _extract_labels(self, metric_name: str) -> Dict[str, str]:
        """Extract labels from metric name."""
        # Simple label extraction - you might want to make this more sophisticated
        if '{' in metric_name and '}' in metric_name:
            label_part = metric_name[metric_name.find('{')+1:metric_name.find('}')]
            labels = {}
            for label_pair in label_part.split(','):
                if '=' in label_pair:
                    key, value = label_pair.split('=', 1)
                    labels[key.strip()] = value.strip()
            return labels
        return {}
    
    async def update_health_metrics(self):
        """Update health metrics from health checker."""
        try:
            health_status = await health_checker.get_health_status()
            
            # Update individual component health
            for check_name, check_result in health_status.get('checks', {}).items():
                health_value = 1 if check_result['status'] == 'healthy' else 0
                self.pipeline_health.labels(component=check_name).set(health_value)
                
        except Exception as e:
            logging.error(f"Error updating health metrics: {e}")
    
    def update_alert_metrics(self):
        """Update alert metrics from alert manager."""
        try:
            # Reset all severity counters
            for severity in ['low', 'medium', 'high', 'critical']:
                self.active_alerts.labels(severity=severity).set(0)
            
            # Count active alerts by severity
            active_alerts = alert_manager.get_active_alerts()
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Update metrics
            for severity, count in severity_counts.items():
                self.active_alerts.labels(severity=severity).set(count)
                
        except Exception as e:
            logging.error(f"Error updating alert metrics: {e}")


class MonitoringDashboard:
    """HTTP dashboard for monitoring endpoints."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize the monitoring dashboard."""
        self.host = host
        self.port = port
        self.app = FastAPI(title="NLP Pipeline Monitoring", version="1.0.0")
        self.prometheus_metrics = PrometheusMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
        
        # Start background metrics updater
        self._metrics_update_task = None
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with available endpoints."""
            return {
                "service": "NLP Pipeline Monitoring",
                "version": "1.0.0",
                "endpoints": {
                    "/health": "Health check endpoint",
                    "/metrics": "Prometheus metrics endpoint",
                    "/metrics/json": "JSON metrics endpoint",
                    "/alerts": "Active alerts",
                    "/alerts/history": "Alert history",
                    "/stats": "System statistics"
                }
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            try:
                health_status = await health_checker.get_health_status()
                status_code = 200 if health_status['status'] == 'healthy' else 503
                return Response(
                    content=json.dumps(health_status, indent=2),
                    status_code=status_code,
                    media_type="application/json"
                )
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return Response(
                    content=json.dumps({"status": "error", "message": str(e)}),
                    status_code=500,
                    media_type="application/json"
                )
        
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus metrics endpoint."""
            try:
                # Update metrics before serving
                self.prometheus_metrics.update_from_metrics_collector()
                await self.prometheus_metrics.update_health_metrics()
                self.prometheus_metrics.update_alert_metrics()
                
                # Generate Prometheus format
                metrics_output = generate_latest(self.prometheus_metrics.registry)
                return PlainTextResponse(
                    content=metrics_output.decode('utf-8'),
                    media_type=CONTENT_TYPE_LATEST
                )
            except Exception as e:
                self.logger.error(f"Metrics endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics/json")
        async def json_metrics():
            """JSON metrics endpoint."""
            try:
                # Get metrics from collector
                collector_metrics = metrics_collector.get_metrics()
                
                # Get health status
                health_status = await health_checker.get_health_status()
                
                # Get alert statistics
                alert_stats = alert_manager.get_alert_statistics()
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": collector_metrics,
                    "health": health_status,
                    "alerts": alert_stats
                }
            except Exception as e:
                self.logger.error(f"JSON metrics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts")
        async def active_alerts():
            """Get active alerts."""
            try:
                alerts = alert_manager.get_active_alerts()
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(alerts),
                    "alerts": alerts
                }
            except Exception as e:
                self.logger.error(f"Active alerts error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts/history")
        async def alert_history(limit: int = 100):
            """Get alert history."""
            try:
                if limit > 1000:
                    limit = 1000  # Cap at 1000 for performance
                
                history = alert_manager.get_alert_history(limit)
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "count": len(history),
                    "alerts": history
                }
            except Exception as e:
                self.logger.error(f"Alert history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def system_stats():
            """Get system statistics."""
            try:
                # Get current metrics
                current_metrics = metrics_collector.get_metrics()
                
                # Get health status
                health_status = await health_checker.get_health_status()
                
                # Get alert statistics
                alert_stats = alert_manager.get_alert_statistics()
                
                # Calculate some derived statistics
                stats = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": {
                        "cpu_percent": current_metrics.get('gauges', {}).get('system_cpu_percent', 0),
                        "memory_percent": current_metrics.get('gauges', {}).get('system_memory_percent', 0),
                        "disk_percent": current_metrics.get('gauges', {}).get('system_disk_percent', 0)
                    },
                    "processing": {
                        "documents_processed": current_metrics.get('counters', {}).get('documents_processed', 0),
                        "processing_errors": current_metrics.get('counters', {}).get('processing_errors', 0)
                    },
                    "health": {
                        "overall_status": health_status.get('status', 'unknown'),
                        "healthy_checks": sum(1 for check in health_status.get('checks', {}).values() 
                                            if check.get('status') == 'healthy'),
                        "total_checks": len(health_status.get('checks', {}))
                    },
                    "alerts": alert_stats,
                    "uptime": "TODO"  # Could track service start time
                }
                
                return stats
            except Exception as e:
                self.logger.error(f"System stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """Start the dashboard server."""
        # Start health checker if not already running
        if not health_checker._running:
            await health_checker.start()
        
        # Start metrics update task
        self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())
        
        # Start the FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        self.logger.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
        await server.serve()
    
    async def stop(self):
        """Stop the dashboard server."""
        if self._metrics_update_task:
            self._metrics_update_task.cancel()
            try:
                await self._metrics_update_task
            except asyncio.CancelledError:
                pass
        
        await health_checker.stop()
    
    async def _metrics_update_loop(self):
        """Background task to update metrics."""
        while True:
            try:
                # Update Prometheus metrics from collector
                self.prometheus_metrics.update_from_metrics_collector()
                await self.prometheus_metrics.update_health_metrics()
                self.prometheus_metrics.update_alert_metrics()
                
                # Wait 30 seconds before next update
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(60)  # Back off on error


# Global dashboard instance
monitoring_dashboard = MonitoringDashboard()


async def start_monitoring_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the monitoring server."""
    dashboard = MonitoringDashboard(host, port)
    await dashboard.start()


if __name__ == "__main__":
    # Allow running the dashboard standalone
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    asyncio.run(start_monitoring_server(host, port))