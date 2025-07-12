"""
Example usage of the monitoring module.

This script demonstrates how to use the various monitoring components:
- Collecting metrics
- Setting up health checks  
- Configuring alerts
- Starting the monitoring dashboard
"""

import asyncio
import time
import random
from datetime import datetime

from src.monitoring import (
    metrics_collector, health_checker, alert_manager, monitoring_dashboard,
    setup_default_alert_rules, AlertRule, AlertSeverity,
    SlackNotificationChannel, EmailNotificationChannel
)


async def simulate_document_processing():
    """Simulate document processing to generate metrics."""
    print("Starting document processing simulation...")
    
    for i in range(100):
        # Simulate processing time
        processing_time = random.uniform(0.5, 3.0)
        
        # Use timer context manager
        with metrics_collector.timer('document_processing_duration', {'model': 'bert'}):
            await asyncio.sleep(processing_time)
        
        # Increment counters
        metrics_collector.increment_counter('documents_processed', labels={'status': 'success'})
        
        # Simulate occasional errors
        if random.random() < 0.1:  # 10% error rate
            metrics_collector.increment_counter('processing_errors', labels={'type': 'timeout'})
        
        # Set gauge for queue size
        queue_size = max(0, 50 - i + random.randint(-10, 10))
        metrics_collector.set_gauge('processing_queue_size', queue_size)
        
        print(f"Processed document {i+1}/100, queue size: {queue_size}")
        
        # Short delay between documents
        await asyncio.sleep(0.1)


def setup_custom_health_checks():
    """Setup custom health checks."""
    print("Setting up custom health checks...")
    
    def check_model_service():
        """Check if model service is responding."""
        # Simulate model service check
        return random.random() > 0.1  # 90% healthy
    
    def check_database():
        """Check database connectivity.""" 
        # Simulate database check
        return random.random() > 0.05  # 95% healthy
    
    # Register custom checks
    health_checker.register_check(
        "model_service_custom",
        check_model_service,
        interval=30,
        critical=True
    )
    
    health_checker.register_check(
        "database_custom", 
        check_database,
        interval=60,
        critical=True
    )


def setup_alert_notifications():
    """Setup alert notification channels."""
    print("Setting up alert notifications...")
    
    # Setup Slack notifications (requires webhook URL)
    # slack_channel = SlackNotificationChannel(
    #     webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    #     channel="#alerts"
    # )
    # alert_manager.add_notification_channel(slack_channel)
    
    # Setup email notifications (requires SMTP configuration)
    # email_channel = EmailNotificationChannel(
    #     smtp_host="smtp.gmail.com",
    #     smtp_port=587,
    #     username="your-email@gmail.com",
    #     password="your-app-password",
    #     from_email="your-email@gmail.com",
    #     to_emails=["admin@company.com"]
    # )
    # alert_manager.add_notification_channel(email_channel)
    
    print("Note: Notification channels are commented out - configure with your credentials")


def setup_custom_alerts():
    """Setup custom alert rules."""
    print("Setting up custom alert rules...")
    
    # Alert for high processing queue
    queue_alert = AlertRule(
        name="high_processing_queue",
        condition=lambda x: x > 30,
        severity=AlertSeverity.MEDIUM,
        message_template="Processing queue is backed up with {value} items",
        source="processing_queue_size",
        threshold=30.0,
        cooldown_period=120
    )
    alert_manager.add_rule(queue_alert)
    
    # Alert for high error rate
    error_rate_alert = AlertRule(
        name="high_error_rate", 
        condition=lambda x: x > 5,  # More than 5 errors
        severity=AlertSeverity.HIGH,
        message_template="Error rate is high: {value} errors detected",
        source="processing_errors",
        threshold=5.0,
        cooldown_period=300
    )
    alert_manager.add_rule(error_rate_alert)


async def monitor_metrics():
    """Monitor and evaluate metrics for alerts."""
    print("Starting metrics monitoring...")
    
    while True:
        try:
            # Get current metrics
            current_metrics = metrics_collector.get_metrics()
            
            # Evaluate gauges for alerts
            for metric_name, value in current_metrics.get('gauges', {}).items():
                await alert_manager.evaluate_metric(metric_name, value)
            
            # Evaluate counters for alerts  
            for metric_name, value in current_metrics.get('counters', {}).items():
                await alert_manager.evaluate_metric(metric_name, value)
            
            # Wait before next evaluation
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Error in metrics monitoring: {e}")
            await asyncio.sleep(60)


async def print_status_updates():
    """Print periodic status updates."""
    while True:
        try:
            # Get current metrics
            metrics = metrics_collector.get_metrics()
            
            # Get health status
            health = await health_checker.get_health_status()
            
            # Get alert stats
            alert_stats = alert_manager.get_alert_statistics()
            
            print(f"\n=== Status Update - {datetime.now().strftime('%H:%M:%S')} ===")
            print(f"Documents processed: {metrics.get('counters', {}).get('documents_processed', 0)}")
            print(f"Processing errors: {metrics.get('counters', {}).get('processing_errors', 0)}")
            print(f"Queue size: {metrics.get('gauges', {}).get('processing_queue_size', 0)}")
            print(f"Health status: {health.get('status', 'unknown')}")
            print(f"Active alerts: {alert_stats.get('active_alerts', 0)}")
            
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"Error printing status: {e}")
            await asyncio.sleep(30)


async def main():
    """Main function to run the monitoring example."""
    print("=== NLP Pipeline Monitoring Example ===\n")
    
    # Setup monitoring components
    setup_custom_health_checks()
    setup_alert_notifications() 
    setup_custom_alerts()
    setup_default_alert_rules()
    
    # Start health checker
    await health_checker.start()
    
    print(f"Monitoring dashboard will be available at: http://localhost:8080")
    print("Available endpoints:")
    print("  - http://localhost:8080/health")
    print("  - http://localhost:8080/metrics")
    print("  - http://localhost:8080/metrics/json") 
    print("  - http://localhost:8080/alerts")
    print("  - http://localhost:8080/stats")
    print()
    
    # Start background tasks
    tasks = [
        asyncio.create_task(simulate_document_processing()),
        asyncio.create_task(monitor_metrics()),
        asyncio.create_task(print_status_updates()),
        asyncio.create_task(monitoring_dashboard.start())
    ]
    
    try:
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nShutting down monitoring example...")
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Stop monitoring components
        await health_checker.stop()
        await monitoring_dashboard.stop()
        
        print("Monitoring example stopped.")


if __name__ == "__main__":
    asyncio.run(main())