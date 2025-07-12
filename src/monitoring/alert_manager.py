"""
Alert manager for handling monitoring alerts and notifications.
"""

import asyncio
import json
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, asdict
import aiohttp

# Optional email imports - will be None if not available
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    print("Warning: Email functionality not available. Install email dependencies if needed.")
    smtplib = None
    MimeText = None
    MimeMultipart = None
    EMAIL_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    timestamp: datetime
    labels: Dict[str, str]
    value: Optional[float] = None
    threshold: Optional[float] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class AlertRule:
    """Defines an alert rule."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity,
        message_template: str,
        source: str,
        labels: Optional[Dict[str, str]] = None,
        threshold: Optional[float] = None,
        cooldown_period: int = 300  # 5 minutes default
    ):
        """
        Initialize an alert rule.
        
        Args:
            name: Alert rule name
            condition: Function that returns True if alert should fire
            severity: Alert severity level
            message_template: Template for alert message
            source: Source component generating the alert
            labels: Additional labels for the alert
            threshold: Threshold value (for documentation)
            cooldown_period: Minimum time between alerts in seconds
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.source = source
        self.labels = labels or {}
        self.threshold = threshold
        self.cooldown_period = cooldown_period
        self.last_fired: Optional[datetime] = None


class NotificationChannel:
    """Base class for notification channels."""
    
    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Returns True if successful."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True
    ):
        """Initialize email notification channel."""
        if not EMAIL_AVAILABLE:
            raise ImportError("Email functionality not available. Check your Python installation.")
        
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
        self.logger = logging.getLogger(__name__)
    
    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            await asyncio.to_thread(self._send_smtp, msg)
            self.logger.info(f"Email alert sent for {alert.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_smtp(self, msg: MimeMultipart):
        """Send email via SMTP."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107", 
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body>
            <h2 style="color: {color};">{alert.name}</h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Status:</strong> {alert.status.value.upper()}</p>
            <p><strong>Source:</strong> {alert.source}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><strong>Message:</strong> {alert.message}</p>
            {f'<p><strong>Value:</strong> {alert.value}</p>' if alert.value is not None else ''}
            {f'<p><strong>Threshold:</strong> {alert.threshold}</p>' if alert.threshold is not None else ''}
            <h3>Labels:</h3>
            <ul>
                {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in alert.labels.items())}
            </ul>
        </body>
        </html>
        """


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        """Initialize webhook notification channel."""
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)
    
    async def send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            payload = alert.to_dict()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook alert sent for {alert.name}")
                        return True
                    else:
                        self.logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        """Initialize Slack notification channel."""
        self.webhook_url = webhook_url
        self.channel = channel
        self.logger = logging.getLogger(__name__)
    
    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            # Map severity to colors
            severity_colors = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            color = severity_colors.get(alert.severity, "good")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.name}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                        ],
                        "footer": "NLP Pipeline Monitoring"
                    }
                ]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            # Add value and threshold if present
            if alert.value is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Value", 
                    "value": str(alert.value), 
                    "short": True
                })
            
            if alert.threshold is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Threshold", 
                    "value": str(alert.threshold), 
                    "short": True
                })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack alert sent for {alert.name}")
                        return True
                    else:
                        self.logger.error(f"Slack webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: List[NotificationChannel] = []
        self.logger = logging.getLogger(__name__)
        self._alert_history: List[Alert] = []
        self._max_history = 1000
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {type(channel).__name__}")
    
    async def evaluate_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Evaluate a metric against all matching alert rules.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            labels: Optional metric labels
        """
        labels = labels or {}
        
        for rule_name, rule in self.rules.items():
            # Check if rule applies to this metric
            if not self._rule_matches_metric(rule, metric_name, labels):
                continue
            
            # Check cooldown period
            if rule.last_fired and datetime.utcnow() - rule.last_fired < timedelta(seconds=rule.cooldown_period):
                continue
            
            # Evaluate condition
            if rule.condition(value):
                await self._fire_alert(rule, value, labels)
            else:
                # Check if we need to resolve an existing alert
                alert_id = self._generate_alert_id(rule.name, labels)
                if alert_id in self.active_alerts:
                    await self._resolve_alert(alert_id)
    
    async def _fire_alert(self, rule: AlertRule, value: float, labels: Dict[str, str]):
        """Fire an alert."""
        alert_id = self._generate_alert_id(rule.name, labels)
        
        # Check if alert is already active
        if alert_id in self.active_alerts:
            return
        
        # Create alert
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=rule.message_template.format(value=value, **labels),
            source=rule.source,
            timestamp=datetime.utcnow(),
            labels=labels,
            value=value,
            threshold=rule.threshold
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self._add_to_history(alert)
        
        # Update rule
        rule.last_fired = datetime.utcnow()
        
        # Send notifications
        await self._send_notifications(alert)
        
        self.logger.warning(f"Alert fired: {rule.name} - {alert.message}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        # Move to history and remove from active
        self._add_to_history(alert)
        del self.active_alerts[alert_id]
        
        # Send resolution notification
        await self._send_notifications(alert)
        
        self.logger.info(f"Alert resolved: {alert.name}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for channel in self.notification_channels:
            try:
                await channel.send(alert)
            except Exception as e:
                self.logger.error(f"Failed to send notification via {type(channel).__name__}: {e}")
    
    def _rule_matches_metric(self, rule: AlertRule, metric_name: str, labels: Dict[str, str]) -> bool:
        """Check if a rule matches a metric."""
        # For now, simple name matching. Could be extended with pattern matching
        return rule.source == metric_name or metric_name.startswith(rule.source)
    
    def _generate_alert_id(self, rule_name: str, labels: Dict[str, str]) -> str:
        """Generate unique alert ID."""
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f"{rule_name}:{hash(label_str)}"
    
    def _add_to_history(self, alert: Alert):
        """Add alert to history."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history.pop(0)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return [alert.to_dict() for alert in self._alert_history[-limit:]]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self._alert_history)
        active_count = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self._alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'severity_breakdown': severity_counts,
            'alert_rules': len(self.rules),
            'notification_channels': len(self.notification_channels)
        }


# Global alert manager instance
alert_manager = AlertManager()

# Pre-configured alert rules for common metrics
def setup_default_alert_rules():
    """Setup default alert rules for common metrics."""
    
    # High CPU usage
    alert_manager.add_rule(AlertRule(
        name="high_cpu_usage",
        condition=lambda x: x > 80,
        severity=AlertSeverity.HIGH,
        message_template="CPU usage is {value:.1f}%, exceeding threshold",
        source="system_cpu_percent",
        threshold=80.0,
        cooldown_period=300
    ))
    
    # High memory usage
    alert_manager.add_rule(AlertRule(
        name="high_memory_usage",
        condition=lambda x: x > 85,
        severity=AlertSeverity.HIGH,
        message_template="Memory usage is {value:.1f}%, exceeding threshold",
        source="system_memory_percent",
        threshold=85.0,
        cooldown_period=300
    ))
    
    # High disk usage
    alert_manager.add_rule(AlertRule(
        name="high_disk_usage",
        condition=lambda x: x > 90,
        severity=AlertSeverity.CRITICAL,
        message_template="Disk usage is {value:.1f}%, exceeding threshold",
        source="system_disk_percent",
        threshold=90.0,
        cooldown_period=600
    ))
    
    # Slow document processing
    alert_manager.add_rule(AlertRule(
        name="slow_document_processing",
        condition=lambda x: x > 5.0,  # 5 seconds
        severity=AlertSeverity.MEDIUM,
        message_template="Document processing time is {value:.2f}s, exceeding threshold",
        source="document_processing_duration",
        threshold=5.0,
        cooldown_period=180
    ))
    
    # High model inference latency
    alert_manager.add_rule(AlertRule(
        name="high_model_latency",
        condition=lambda x: x > 2.0,  # 2 seconds
        severity=AlertSeverity.MEDIUM,
        message_template="Model inference latency is {value:.2f}s, exceeding threshold",
        source="model_inference_duration",
        threshold=2.0,
        cooldown_period=180
    ))
    
    # Low document processing throughput
    alert_manager.add_rule(AlertRule(
        name="low_processing_throughput",
        condition=lambda x: x < 10,  # Less than 10 docs per minute
        severity=AlertSeverity.MEDIUM,
        message_template="Document processing throughput is {value:.1f} docs/min, below threshold",
        source="documents_processed_per_minute",
        threshold=10.0,
        cooldown_period=300
    ))