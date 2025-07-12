"""
Metrics collector for monitoring document processing performance.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict, deque
from contextlib import contextmanager


class MetricsCollector:
    """Collects and stores performance metrics for the NLP pipeline."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of metric entries to keep in memory
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Start system metrics collection
        self._system_monitor_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._system_monitor_thread.start()
    
    @contextmanager
    def timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_histogram(metric_name, duration, labels)
    
    def increment_counter(self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            value: Value to increment by
            labels: Optional labels for the metric
        """
        key = self._generate_key(metric_name, labels)
        with self._lock:
            self._counters[key] += value
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Name of the gauge
            value: Current value
            labels: Optional labels for the metric
        """
        key = self._generate_key(metric_name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a value in a histogram.
        
        Args:
            metric_name: Name of the histogram
            value: Value to record
            labels: Optional labels for the metric
        """
        key = self._generate_key(metric_name, labels)
        with self._lock:
            if len(self._histograms[key]) >= self.max_history:
                self._histograms[key].pop(0)
            self._histograms[key].append(value)
            
            # Also store timestamped entry
            self._metrics[key].append({
                'timestamp': datetime.utcnow(),
                'value': value,
                'type': 'histogram'
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            metrics = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {}
            }
            
            # Calculate histogram statistics
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    metrics['histograms'][key] = {
                        'count': len(values),
                        'sum': sum(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'p50': self._percentile(sorted_values, 0.5),
                        'p90': self._percentile(sorted_values, 0.9),
                        'p95': self._percentile(sorted_values, 0.95),
                        'p99': self._percentile(sorted_values, 0.99)
                    }
            
            return metrics
    
    def get_metric_history(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Get historical values for a specific metric.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels for the metric
            
        Returns:
            List of historical metric entries
        """
        key = self._generate_key(metric_name, labels)
        with self._lock:
            return list(self._metrics.get(key, []))
    
    def _collect_system_metrics(self):
        """Continuously collect system metrics."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge('system_cpu_percent', cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.set_gauge('system_memory_used_bytes', memory.used)
                self.set_gauge('system_memory_percent', memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.set_gauge('system_disk_used_bytes', disk.used)
                self.set_gauge('system_disk_percent', disk.percent)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.set_gauge('system_network_bytes_sent', net_io.bytes_sent)
                self.set_gauge('system_network_bytes_recv', net_io.bytes_recv)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                time.sleep(30)  # Back off on error
    
    def _generate_key(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return metric_name
        
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f"{metric_name}{{{label_str}}}"
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = int(len(sorted_values) * percentile)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        
        return sorted_values[index]


# Global metrics collector instance
metrics_collector = MetricsCollector()