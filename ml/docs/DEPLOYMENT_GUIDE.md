# Deployment Guide - Collaborative Filtering System

## Overview

This guide provides comprehensive instructions for deploying the collaborative filtering system in production environments, covering infrastructure setup, configuration management, monitoring, and maintenance procedures.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration Management](#configuration-management)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Performance Optimization](#performance-optimization)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling Strategies](#scaling-strategies)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.4 GHz
- **Memory**: 8 GB RAM
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 20.04 LTS or CentOS 8

#### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **Memory**: 32 GB RAM
- **Storage**: 200 GB NVMe SSD
- **Network**: 10 Gbps
- **OS**: Ubuntu 22.04 LTS

### Software Dependencies

#### Core Dependencies
- Python 3.9+
- Redis 6.0+
- PostgreSQL 13+ (for data storage)
- nginx (for load balancing)

#### Python Packages
```bash
pip install -r requirements.txt
```

Required packages:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pandas==2.1.4
- numpy==1.24.4
- scikit-learn==1.3.2
- redis==5.0.1
- pydantic==2.5.0

## Environment Setup

### Development Environment

```bash
# Clone repository
git clone <repository-url>
cd Timbre/ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_URL="redis://localhost:6379"
export DEBUG=true

# Run development server
python main.py
```

### Staging Environment

```bash
# Set staging configuration
export ENVIRONMENT=staging
export REDIS_URL="redis://staging-redis:6379"
export DEBUG=false
export API_WORKERS=2

# Run with monitoring
python main.py --host 0.0.0.0 --port 8001
```

### Production Environment

```bash
# Production configuration
export ENVIRONMENT=production
export REDIS_URL="redis://prod-redis-cluster:6379"
export DEBUG=false
export API_WORKERS=8
export LOG_LEVEL=INFO

# Run with process manager
gunicorn main:app --workers 8 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

## Configuration Management

### Environment Variables

Create environment-specific configuration files:

#### `.env.development`
```bash
# Development Configuration
DEBUG=true
API_HOST=127.0.0.1
API_PORT=8001
API_WORKERS=1

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Model Configuration
NMF_N_COMPONENTS=50
NMF_RANDOM_STATE=42

# Logging
LOG_LEVEL=DEBUG
```

#### `.env.production`
```bash
# Production Configuration
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=8

# Redis Configuration (Upstash or cluster)
REDIS_URL=redis://username:password@redis-cluster:6379
REDIS_DB=0

# Model Configuration
NMF_N_COMPONENTS=100
NMF_RANDOM_STATE=42

# Logging
LOG_LEVEL=INFO

# Security
MAX_REQUEST_SIZE=10MB
RATE_LIMIT=100/minute
```

### Configuration Loading

```python
# config/deployment.py
import os
from typing import Optional
from pydantic_settings import BaseSettings

class DeploymentSettings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_WORKERS: int = 4
    
    # Security
    MAX_REQUEST_SIZE: str = "10MB"
    RATE_LIMIT: str = "100/minute"
    
    # Performance
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = f".env.{os.getenv('ENVIRONMENT', 'development')}"
        case_sensitive = True

# Load configuration
deployment_config = DeploymentSettings()
```

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/ping')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
      - DEBUG=false
      - API_WORKERS=4
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ml-service
    restart: unless-stopped

volumes:
  redis_data:
```

### Build and Deploy

```bash
# Build images
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f ml-service

# Scale ML service
docker-compose up -d --scale ml-service=3
```

## Production Deployment

### Kubernetes Deployment

#### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: timbral-ml-service
  labels:
    app: timbral-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: timbral-ml
  template:
    metadata:
      labels:
        app: timbral-ml
    spec:
      containers:
      - name: ml-service
        image: timbral/ml-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: timbral-secrets
              key: redis-url
        - name: API_WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /ping
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ping
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: logs
        emptyDir: {}
```

#### Service Manifest

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: timbral-ml-service
spec:
  selector:
    app: timbral-ml
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
```

#### Ingress Manifest

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: timbral-ml-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ml-api.timbrality.com
    secretName: timbral-ml-tls
  rules:
  - host: ml-api.timbrality.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: timbral-ml-service
            port:
              number: 8001
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment timbral-ml-service --replicas=5

# Update deployment
kubectl set image deployment/timbral-ml-service ml-service=timbral/ml-service:v2.0

# Rolling update
kubectl rollout status deployment/timbral-ml-service
kubectl rollout undo deployment/timbral-ml-service  # Rollback if needed
```

## Monitoring and Logging

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

def setup_logging():
    # Create formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = RotatingFileHandler(
        'logs/timbral-ml.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('timbral_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('timbral_request_duration_seconds', 'Request latency')
ACTIVE_MODELS = Gauge('timbral_active_models', 'Number of active models')
CACHE_HIT_RATE = Gauge('timbral_cache_hit_rate', 'Cache hit rate')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Process request
        await self.app(scope, receive, send)
        
        # Record metrics
        REQUEST_COUNT.labels(method=method, endpoint=path).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

# Start metrics server
def start_metrics_server(port=9090):
    start_http_server(port)
```

### Health Checks

```python
# health.py
from fastapi import APIRouter
from typing import Dict, Any
import psutil
import time

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "timbral-ml",
        "version": "1.0.0",
        "checks": {
            "database": await check_database(),
            "redis": await check_redis(),
            "models": await check_models(),
            "system": check_system_resources()
        }
    }

async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        # Database health check logic
        return {"status": "healthy", "latency_ms": 5}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        # Redis health check logic
        return {"status": "healthy", "latency_ms": 2}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
```

## Performance Optimization

### Model Loading Optimization

```python
# model_manager.py
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

class ModelManager:
    def __init__(self):
        self.model_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def load_model_async(self, model_path: str) -> NMFModel:
        """Load model asynchronously to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._load_model_sync,
            model_path
        )
    
    def _load_model_sync(self, model_path: str) -> NMFModel:
        """Synchronous model loading."""
        if model_path not in self.model_cache:
            model = NMFModel()
            model.load(model_path)
            self.model_cache[model_path] = model
        return self.model_cache[model_path]
```

### Recommendation Caching Strategy

```python
# caching.py
from functools import wraps
import hashlib
import json

def cache_recommendations(ttl: int = 3600):
    """Decorator for caching recommendations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"rec:{hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()}"
            
            # Try to get from cache
            cached = redis_connector.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Generate recommendations
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_connector.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

@cache_recommendations(ttl=1800)
async def get_user_recommendations(user_id: int, top_k: int = 10):
    """Cached recommendation generation."""
    # Recommendation logic here
    pass
```

### Connection Pooling

```python
# redis_pool.py
import redis.asyncio as redis
from redis.connection import ConnectionPool

class RedisManager:
    def __init__(self):
        self.pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=20,
            retry_on_timeout=True,
            health_check_interval=30
        )
        self.redis = redis.Redis(connection_pool=self.pool)
    
    async def get_client(self):
        return self.redis
    
    async def close(self):
        await self.redis.close()
        await self.pool.disconnect()
```

## Backup and Recovery

### Model Backup Strategy

```python
# backup.py
import boto3
import os
from datetime import datetime
import shutil

class ModelBackupManager:
    def __init__(self, bucket_name: str):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
    
    def backup_model(self, model_path: str, model_name: str):
        """Backup model to S3."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_key = f"models/{model_name}/{timestamp}/model.pkl"
        
        try:
            self.s3_client.upload_file(
                model_path,
                self.bucket_name,
                backup_key
            )
            return backup_key
        except Exception as e:
            raise Exception(f"Backup failed: {e}")
    
    def restore_model(self, backup_key: str, local_path: str):
        """Restore model from S3."""
        try:
            self.s3_client.download_file(
                self.bucket_name,
                backup_key,
                local_path
            )
            return local_path
        except Exception as e:
            raise Exception(f"Restore failed: {e}")
    
    def list_backups(self, model_name: str):
        """List available backups for a model."""
        prefix = f"models/{model_name}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        return [obj['Key'] for obj in response.get('Contents', [])]
```

### Database Backup

```bash
#!/bin/bash
# backup_db.sh

# Configuration
DB_NAME="timbral_db"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/db_backup_${DATE}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp "${BACKUP_FILE}.gz" s3://timbral-backups/database/

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream timbral_ml {
    least_conn;
    server ml-service-1:8001 weight=1 max_fails=3 fail_timeout=30s;
    server ml-service-2:8001 weight=1 max_fails=3 fail_timeout=30s;
    server ml-service-3:8001 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name ml-api.timbrality.com;
    
    location / {
        proxy_pass http://timbral_ml;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://timbral_ml;
    }
}
```

#### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: timbral-ml-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: timbral-ml-service
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

### Vertical Scaling

```yaml
# k8s/vertical-pod-autoscaler.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: timbral-ml-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: timbral-ml-service
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: ml-service
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
```

## Troubleshooting

### Common Issues

#### High Memory Usage

**Symptoms:**
- Out of memory errors
- Slow response times
- Pod restarts

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods
docker stats

# Check memory allocation
kubectl describe pod <pod-name>
```

**Solutions:**
- Increase memory limits
- Optimize model size
- Implement model unloading
- Use memory profiling

#### Redis Connection Issues

**Symptoms:**
- Cache misses
- Connection timeouts
- Recommendation delays

**Diagnosis:**
```bash
# Test Redis connectivity
redis-cli ping

# Check Redis metrics
redis-cli info memory
redis-cli info clients
```

**Solutions:**
- Check Redis server status
- Verify network connectivity
- Adjust connection pool settings
- Implement circuit breakers

#### Model Loading Failures

**Symptoms:**
- Model not found errors
- Pickle loading errors
- Inconsistent predictions

**Diagnosis:**
```python
# Verify model file
import os
print(os.path.exists("models/production_model.pkl"))

# Check model integrity
import joblib
try:
    model_data = joblib.load("models/production_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
```

**Solutions:**
- Verify model file existence
- Check file permissions
- Validate model format
- Restore from backup

### Debugging Tools

#### Log Analysis

```bash
# Kubernetes logs
kubectl logs -f deployment/timbral-ml-service

# Docker logs
docker logs -f timbral-ml-service

# Log aggregation with grep
kubectl logs deployment/timbral-ml-service | grep ERROR

# Structured log querying (if using JSON logs)
kubectl logs deployment/timbral-ml-service | jq '.level == "ERROR"'
```

#### Performance Profiling

```python
# performance_profiler.py
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper

@profile_function
def generate_recommendations(user_id: int):
    # Function to profile
    pass
```

#### Health Check Script

```python
# health_check.py
import requests
import sys
import time

def check_service_health(base_url: str):
    """Comprehensive service health check."""
    checks = {
        "ping": f"{base_url}/ping",
        "health": f"{base_url}/health",
        "metrics": f"{base_url}/metrics"
    }
    
    results = {}
    for check_name, url in checks.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            latency = time.time() - start_time
            
            results[check_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 2)
            }
        except Exception as e:
            results[check_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return results

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    results = check_service_health(base_url)
    
    print("Service Health Check Results:")
    for check, result in results.items():
        print(f"  {check}: {result}")
    
    # Exit with error code if any check failed
    if any(r["status"] == "unhealthy" for r in results.values()):
        sys.exit(1)
```

This comprehensive deployment guide provides the foundation for successfully deploying and maintaining the collaborative filtering system in production environments, ensuring reliability, scalability, and performance.