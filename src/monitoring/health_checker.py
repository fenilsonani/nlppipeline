"""
Health checker for monitoring system and service health.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import logging


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Represents a single health check."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: int = 30,
        timeout: int = 10,
        critical: bool = True
    ):
        """
        Initialize a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            interval: Check interval in seconds
            timeout: Check timeout in seconds
            critical: Whether this check is critical for overall health
        """
        self.name = name
        self.check_func = check_func
        self.interval = interval
        self.timeout = timeout
        self.critical = critical
        self.last_check: Optional[datetime] = None
        self.last_status: Optional[bool] = None
        self.consecutive_failures: int = 0


class HealthChecker:
    """Manages system health checks."""
    
    def __init__(self):
        """Initialize the health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self._running = False
        self._check_tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: int = 30,
        timeout: int = 10,
        critical: bool = True
    ):
        """
        Register a new health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            interval: Check interval in seconds
            timeout: Check timeout in seconds
            critical: Whether this check is critical for overall health
        """
        self.checks[name] = HealthCheck(name, check_func, interval, timeout, critical)
    
    async def start(self):
        """Start running health checks."""
        if self._running:
            return
        
        self._running = True
        
        # Register default checks
        self._register_default_checks()
        
        # Start check tasks
        for check in self.checks.values():
            task = asyncio.create_task(self._run_check_loop(check))
            self._check_tasks.append(task)
    
    async def stop(self):
        """Stop running health checks."""
        self._running = False
        
        # Cancel all check tasks
        for task in self._check_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self._check_tasks, return_exceptions=True)
        self._check_tasks.clear()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Dictionary containing overall health status and individual check results
        """
        check_results = {}
        critical_healthy = True
        non_critical_healthy = True
        
        for name, check in self.checks.items():
            if check.last_check is None:
                status = "unknown"
            elif check.last_status:
                status = "healthy"
            else:
                status = "unhealthy"
                if check.critical:
                    critical_healthy = False
                else:
                    non_critical_healthy = False
            
            check_results[name] = {
                'status': status,
                'last_check': check.last_check.isoformat() if check.last_check else None,
                'consecutive_failures': check.consecutive_failures,
                'critical': check.critical
            }
        
        # Determine overall status
        if critical_healthy and non_critical_healthy:
            overall_status = HealthStatus.HEALTHY
        elif critical_healthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        return {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': check_results
        }
    
    async def _run_check_loop(self, check: HealthCheck):
        """Run a health check in a loop."""
        while self._running:
            try:
                # Run the check with timeout
                result = await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(check.check_func)),
                    timeout=check.timeout
                )
                
                check.last_check = datetime.utcnow()
                check.last_status = result
                
                if result:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                    self.logger.warning(f"Health check '{check.name}' failed")
                
            except asyncio.TimeoutError:
                check.consecutive_failures += 1
                check.last_status = False
                self.logger.error(f"Health check '{check.name}' timed out")
                
            except Exception as e:
                check.consecutive_failures += 1
                check.last_status = False
                self.logger.error(f"Health check '{check.name}' error: {e}")
            
            # Wait for next check
            await asyncio.sleep(check.interval)
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # Database connectivity check
        def check_database():
            # Placeholder - implement actual database check
            return True
        
        self.register_check(
            "database",
            check_database,
            interval=30,
            critical=True
        )
        
        # Model service check
        def check_model_service():
            # Placeholder - implement actual model service check
            return True
        
        self.register_check(
            "model_service",
            check_model_service,
            interval=60,
            critical=True
        )
        
        # Storage check
        def check_storage():
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 90  # Healthy if less than 90% full
        
        self.register_check(
            "storage",
            check_storage,
            interval=300,  # Every 5 minutes
            critical=False
        )
        
        # Memory check
        def check_memory():
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 85  # Healthy if less than 85% used
        
        self.register_check(
            "memory",
            check_memory,
            interval=60,
            critical=False
        )


# Global health checker instance
health_checker = HealthChecker()