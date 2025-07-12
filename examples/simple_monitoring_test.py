#!/usr/bin/env python3
"""
Simple test to verify monitoring module functionality.
"""

import asyncio
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring import (
    metrics_collector, health_checker, alert_manager,
    setup_default_alert_rules, AlertRule, AlertSeverity
)


async def test_metrics_collection():
    """Test metrics collection functionality."""
    print("Testing metrics collection...")
    
    # Test counter
    metrics_collector.increment_counter('test_counter', 5)
    metrics_collector.increment_counter('test_counter', 3)
    
    # Test gauge
    metrics_collector.set_gauge('test_gauge', 42.5)
    
    # Test histogram
    metrics_collector.record_histogram('test_histogram', 1.5)
    metrics_collector.record_histogram('test_histogram', 2.0)
    metrics_collector.record_histogram('test_histogram', 0.8)
    
    # Test timer context manager
    with metrics_collector.timer('test_timer'):
        time.sleep(0.1)
    
    # Get metrics
    metrics = metrics_collector.get_metrics()
    
    # Verify results
    assert metrics['counters']['test_counter'] == 8
    assert metrics['gauges']['test_gauge'] == 42.5
    assert metrics['histograms']['test_histogram']['count'] >= 3
    assert 'test_timer' in metrics['histograms']
    
    print("✓ Metrics collection test passed")


async def test_health_checker():
    """Test health checker functionality."""
    print("Testing health checker...")
    
    # Register a custom health check
    def custom_check():
        return True  # Always healthy for test
    
    health_checker.register_check(
        "test_service",
        custom_check,
        interval=5,
        critical=True
    )
    
    # Start health checker
    await health_checker.start()
    
    # Wait a moment for checks to run
    await asyncio.sleep(1)
    
    # Get health status
    health_status = await health_checker.get_health_status()
    
    # Verify results
    assert 'test_service' in health_status['checks']
    print(f"Health status: {health_status['status']}")
    
    await health_checker.stop()
    print("✓ Health checker test passed")


async def test_alert_manager():
    """Test alert manager functionality."""
    print("Testing alert manager...")
    
    # Setup default rules
    setup_default_alert_rules()
    
    # Add a custom rule
    test_rule = AlertRule(
        name="test_alert",
        condition=lambda x: x > 100,
        severity=AlertSeverity.HIGH,
        message_template="Test value is {value}, exceeding threshold",
        source="test_metric",
        threshold=100.0
    )
    alert_manager.add_rule(test_rule)
    
    # Test metric that should trigger alert
    await alert_manager.evaluate_metric("test_metric", 150.0)
    
    # Check for active alerts
    active_alerts = alert_manager.get_active_alerts()
    assert len(active_alerts) > 0
    
    # Get alert statistics
    stats = alert_manager.get_alert_statistics()
    assert stats['active_alerts'] > 0
    
    print(f"Active alerts: {len(active_alerts)}")
    print("✓ Alert manager test passed")


async def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    
    # Simulate some processing with metrics
    for i in range(5):
        metrics_collector.increment_counter('documents_processed')
        metrics_collector.record_histogram('processing_time', 0.5 + i * 0.1)
        
        # Evaluate metrics for alerts
        await alert_manager.evaluate_metric('documents_processed', i + 1)
    
    # Check final state
    metrics = metrics_collector.get_metrics()
    alerts = alert_manager.get_active_alerts()
    
    print(f"Documents processed: {metrics['counters'].get('documents_processed', 0)}")
    print(f"Active alerts: {len(alerts)}")
    print("✓ Integration test passed")


async def main():
    """Run all tests."""
    print("=== Monitoring Module Test Suite ===\n")
    
    try:
        await test_metrics_collection()
        await test_health_checker()
        await test_alert_manager()
        await test_integration()
        
        print("\n=== All Tests Passed! ===")
        print("The monitoring module is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)