# Deployment Guide

## Overview

This guide covers deployment strategies for the NLP Pipeline across different environments, from local development to production Kubernetes clusters. The system is designed for cloud-native deployment with horizontal scalability and high availability.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platforms](#cloud-platforms)
- [Environment Configuration](#environment-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Security Configuration](#security-configuration)
- [Scaling Guidelines](#scaling-guidelines)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 8 GB RAM
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps
- **Python**: 3.8+

#### Recommended Production Requirements
- **CPU**: 16+ cores
- **Memory**: 32+ GB RAM
- **Storage**: 200+ GB NVMe SSD
- **Network**: 10 Gbps
- **GPU**: NVIDIA V100/A100 (optional, for model acceleration)

### Software Dependencies

```bash
# Core dependencies
docker >= 20.10
docker-compose >= 1.29
kubectl >= 1.21
helm >= 3.7

# For Kubernetes deployment
kubernetes >= 1.21
nginx-ingress-controller
cert-manager
prometheus-operator
```

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/fenilsonani/nlp-pipeline.git
cd nlp-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_models.txt

# Set environment variables
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export MODEL_CACHE_SIZE=2

# Run the application
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Development with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Development Environment Variables

Create a `.env` file:

```bash
# Application
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Configuration
MODEL_CACHE_SIZE=2
MODEL_CACHE_TTL=3600
MODEL_DEVICE=cpu

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/nlp_pipeline_dev
REDIS_URL=redis://localhost:6379/0

# Kafka (optional for development)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_ENABLE_STREAMING=false
```

## Docker Deployment

### Single Container Deployment

```bash
# Build the image
docker build -t nlp-pipeline:latest .

# Run with basic configuration
docker run -d \
  --name nlp-pipeline \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://user:pass@db:5432/nlp_pipeline \
  nlp-pipeline:latest
```

### Multi-Container with Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://nlp_user:${DB_PASSWORD}@db:5432/nlp_pipeline
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - db
      - redis
      - kafka
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build: .
    command: ["python", "-m", "src.workers.batch_processor"]
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://nlp_user:${DB_PASSWORD}@db:5432/nlp_pipeline
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - db
      - redis
      - kafka
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      replicas: 3

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=nlp_pipeline
      - POSTGRES_USER=nlp_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U nlp_user"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    volumes:
      - kafka_data:/var/lib/kafka/data
    restart: unless-stopped

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./docker/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
```

### Running Production Docker Compose

```bash
# Set environment variables
export DB_PASSWORD=your_secure_password

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=5

# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Stop services
docker-compose -f docker-compose.prod.yml down
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://get.helm.sh/helm-v3.7.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/

# Verify installation
kubectl version --client
helm version
```

### Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nlp-pipeline
  labels:
    name: nlp-pipeline
```

```bash
kubectl apply -f k8s/namespace.yaml
```

### ConfigMaps and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nlp-pipeline-config
  namespace: nlp-pipeline
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  MODEL_CACHE_SIZE: "5"
  MODEL_CACHE_TTL: "3600"
  REDIS_URL: "redis://redis:6379/0"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nlp-pipeline-secrets
  namespace: nlp-pipeline
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  JWT_SECRET: <base64-encoded-jwt-secret>
  API_SECRET_KEY: <base64-encoded-api-secret>
```

```bash
# Create secrets
kubectl create secret generic nlp-pipeline-secrets \
  --from-literal=DATABASE_URL=postgresql://user:pass@postgres:5432/nlp_pipeline \
  --from-literal=JWT_SECRET=your-jwt-secret \
  --from-literal=API_SECRET_KEY=your-api-secret \
  -n nlp-pipeline
```

### Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: nlp-pipeline
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: nlp_pipeline
        - name: POSTGRES_USER
          value: nlp_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: nlp-pipeline
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-pipeline-api
  namespace: nlp-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlp-pipeline-api
  template:
    metadata:
      labels:
        app: nlp-pipeline-api
    spec:
      containers:
      - name: api
        image: nlp-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nlp-pipeline-secrets
              key: DATABASE_URL
        envFrom:
        - configMapRef:
            name: nlp-pipeline-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: nlp-pipeline-api
  namespace: nlp-pipeline
spec:
  selector:
    app: nlp-pipeline-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nlp-pipeline-ingress
  namespace: nlp-pipeline
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.nlp-pipeline.com
    secretName: nlp-pipeline-tls
  rules:
  - host: api.nlp-pipeline.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nlp-pipeline-api
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nlp-pipeline-hpa
  namespace: nlp-pipeline
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nlp-pipeline-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n nlp-pipeline
kubectl get services -n nlp-pipeline
kubectl get ingress -n nlp-pipeline

# View logs
kubectl logs -f deployment/nlp-pipeline-api -n nlp-pipeline

# Scale deployment
kubectl scale deployment nlp-pipeline-api --replicas=5 -n nlp-pipeline
```

## Cloud Platforms

### AWS EKS Deployment

#### Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

#### Create EKS Cluster

```yaml
# eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: nlp-pipeline-cluster
  region: us-west-2
  version: "1.21"

nodeGroups:
  - name: api-nodes
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 100
    ssh:
      allow: true
    tags:
      Environment: production
      Project: nlp-pipeline
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true

  - name: worker-nodes
    instanceType: c5.2xlarge
    desiredCapacity: 2
    minSize: 1
    maxSize: 5
    volumeSize: 200
    ssh:
      allow: true
    tags:
      Environment: production
      Project: nlp-pipeline
      Role: worker

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
```

```bash
# Create cluster
eksctl create cluster -f eks-cluster.yaml

# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=nlp-pipeline-cluster
```

#### RDS Database Setup

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier nlp-pipeline-db \
  --db-instance-class db.r5.large \
  --engine postgres \
  --engine-version 14.6 \
  --master-username nlp_admin \
  --master-user-password YourSecurePassword \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name default \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted
```

### Google GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create nlp-pipeline-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --disk-size 100GB \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials nlp-pipeline-cluster --zone us-central1-a
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name nlp-pipeline-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group nlp-pipeline-rg \
  --name nlp-pipeline-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group nlp-pipeline-rg --name nlp-pipeline-cluster
```

## Environment Configuration

### Production Environment Variables

```bash
# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# Model Configuration
MODEL_CACHE_SIZE=10
MODEL_CACHE_TTL=7200
MODEL_DEVICE=cuda
MODEL_PRECISION=fp16

# Database Configuration
DATABASE_URL=postgresql://user:pass@postgres-cluster:5432/nlp_pipeline
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache Configuration
REDIS_URL=redis://redis-cluster:6379/0
REDIS_POOL_SIZE=10

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-cluster:9092
KAFKA_GROUP_ID=nlp-pipeline-prod
KAFKA_AUTO_OFFSET_RESET=latest

# Monitoring Configuration
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn

# Security Configuration
JWT_SECRET=your-jwt-secret
API_SECRET_KEY=your-api-secret
CORS_ORIGINS=https://your-frontend.com
```

### Staging Environment Variables

```bash
# Application Configuration
ENVIRONMENT=staging
LOG_LEVEL=DEBUG
DEBUG=true

# API Configuration
API_WORKERS=2
API_TIMEOUT=60

# Model Configuration
MODEL_CACHE_SIZE=3
MODEL_DEVICE=cpu

# Database Configuration
DATABASE_URL=postgresql://user:pass@postgres-staging:5432/nlp_pipeline_staging

# Monitoring Configuration
PROMETHEUS_ENABLED=false
JAEGER_ENABLED=false
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'nlp-pipeline'
      static_configs:
      - targets: ['nlp-pipeline-api.nlp-pipeline:80']
      metrics_path: /metrics
      scrape_interval: 15s
    
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "NLP Pipeline Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, api_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nlp-pipeline-netpol
  namespace: nlp-pipeline
spec:
  podSelector:
    matchLabels:
      app: nlp-pipeline-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Policy

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: nlp-pipeline-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Scaling Guidelines

### Vertical Scaling

```bash
# Increase CPU and memory limits
kubectl patch deployment nlp-pipeline-api -n nlp-pipeline -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "requests": {"memory": "4Gi", "cpu": "2"},
            "limits": {"memory": "8Gi", "cpu": "4"}
          }
        }]
      }
    }
  }
}'
```

### Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment nlp-pipeline-api --replicas=10 -n nlp-pipeline

# Update HPA for automatic scaling
kubectl patch hpa nlp-pipeline-hpa -n nlp-pipeline -p '{
  "spec": {
    "maxReplicas": 50,
    "metrics": [{
      "type": "Resource",
      "resource": {
        "name": "cpu",
        "target": {
          "type": "Utilization",
          "averageUtilization": 60
        }
      }
    }]
  }
}'
```

### Database Scaling

#### Read Replicas

```bash
# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier nlp-pipeline-read-replica \
  --source-db-instance-identifier nlp-pipeline-db \
  --db-instance-class db.r5.large
```

#### Connection Pooling

```yaml
# k8s/pgbouncer.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: nlp-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:latest
        env:
        - name: DATABASES_HOST
          value: postgres
        - name: DATABASES_PORT
          value: "5432"
        - name: DATABASES_USER
          value: nlp_user
        - name: DATABASES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: POOL_MODE
          value: transaction
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "25"
        ports:
        - containerPort: 5432
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n nlp-pipeline

# Describe pod for events
kubectl describe pod <pod-name> -n nlp-pipeline

# Check logs
kubectl logs <pod-name> -n nlp-pipeline

# Check previous container logs
kubectl logs <pod-name> -n nlp-pipeline --previous
```

#### Resource Issues

```bash
# Check resource usage
kubectl top pods -n nlp-pipeline
kubectl top nodes

# Check resource quotas
kubectl describe resourcequota -n nlp-pipeline

# Check limits and requests
kubectl describe pod <pod-name> -n nlp-pipeline | grep -A 5 -B 5 resources
```

#### Networking Issues

```bash
# Test service connectivity
kubectl exec -it <pod-name> -n nlp-pipeline -- curl http://postgres:5432

# Check DNS resolution
kubectl exec -it <pod-name> -n nlp-pipeline -- nslookup postgres

# Check network policies
kubectl get networkpolicy -n nlp-pipeline
```

#### Database Connection Issues

```bash
# Test database connection
kubectl exec -it <api-pod> -n nlp-pipeline -- python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@postgres:5432/nlp_pipeline')
print('Connected successfully')
"

# Check database logs
kubectl logs <postgres-pod> -n nlp-pipeline
```

### Performance Troubleshooting

#### High CPU Usage

```bash
# Check CPU metrics
kubectl top pods -n nlp-pipeline

# Profile application
kubectl exec -it <pod-name> -n nlp-pipeline -- python -m cProfile -o profile.out main.py

# Scale up resources
kubectl patch deployment nlp-pipeline-api -n nlp-pipeline -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "limits": {"cpu": "4"}
          }
        }]
      }
    }
  }
}'
```

#### Memory Issues

```bash
# Check memory usage
kubectl top pods -n nlp-pipeline

# Check for OOMKilled pods
kubectl get pods -n nlp-pipeline | grep OOMKilled

# Increase memory limits
kubectl patch deployment nlp-pipeline-api -n nlp-pipeline -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "limits": {"memory": "8Gi"}
          }
        }]
      }
    }
  }
}'
```

### Health Check Scripts

```bash
#!/bin/bash
# health-check.sh

echo "Checking NLP Pipeline Health..."

# Check API health
API_HEALTH=$(curl -s http://api.nlp-pipeline.com/health | jq -r '.data.status')
echo "API Status: $API_HEALTH"

# Check database connectivity
DB_HEALTH=$(kubectl exec -n nlp-pipeline deployment/nlp-pipeline-api -- python -c "
import psycopg2
try:
    conn = psycopg2.connect('$DATABASE_URL')
    print('healthy')
except:
    print('unhealthy')
")
echo "Database Status: $DB_HEALTH"

# Check model loading
MODEL_HEALTH=$(curl -s http://api.nlp-pipeline.com/models | jq -r '.data.models | length')
echo "Loaded Models: $MODEL_HEALTH"

# Overall status
if [[ "$API_HEALTH" == "healthy" && "$DB_HEALTH" == "healthy" && "$MODEL_HEALTH" -gt 0 ]]; then
    echo "✓ All systems healthy"
    exit 0
else
    echo "✗ System issues detected"
    exit 1
fi
```

This deployment guide provides comprehensive instructions for deploying the NLP Pipeline across various environments and platforms, ensuring scalability, security, and observability in production environments.