# Performance Optimization Guide

## Overview

This guide provides comprehensive strategies and techniques for optimizing the performance of the NLP Pipeline. The goal is to achieve sub-200ms response times while maintaining high throughput and low resource consumption.

## Table of Contents

- [Performance Targets](#performance-targets)
- [System Architecture Optimizations](#system-architecture-optimizations)
- [Model Optimization](#model-optimization)
- [Caching Strategies](#caching-strategies)
- [Database Optimization](#database-optimization)
- [Network Optimization](#network-optimization)
- [Memory Management](#memory-management)
- [CPU Optimization](#cpu-optimization)
- [Monitoring and Profiling](#monitoring-and-profiling)
- [Load Testing](#load-testing)
- [Best Practices](#best-practices)

## Performance Targets

### Response Time Targets

| Operation | Target (ms) | Acceptable (ms) | Max (ms) |
|-----------|-------------|-----------------|----------|
| Health Check | 5 | 10 | 50 |
| Sentiment Analysis (single) | 15 | 25 | 100 |
| Sentiment Analysis (batch-32) | 150 | 200 | 500 |
| Entity Extraction (single) | 25 | 50 | 150 |
| Entity Extraction (batch-16) | 180 | 250 | 600 |
| Combined Analysis | 40 | 75 | 200 |

### Throughput Targets

| Operation | Target (req/s) | Min (req/s) |
|-----------|----------------|-------------|
| Single Text Analysis | 100 | 50 |
| Batch Processing | 300 | 150 |
| Concurrent Users | 1000 | 500 |

### Resource Targets

| Resource | Target | Max |
|----------|--------|-----|
| CPU Usage | 70% | 85% |
| Memory Usage | 75% | 90% |
| GPU Usage | 80% | 95% |
| Disk I/O | 60% | 80% |

## System Architecture Optimizations

### Asynchronous Processing

```python
# src/api/async_handler.py
import asyncio
import aiohttp
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class AsyncNLPHandler:
    """Asynchronous NLP processing handler for better concurrency."""
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_text_async(self, text: str, model_type: str) -> Dict[str, Any]:
        """Process text asynchronously."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._sync_process,
                text,
                model_type
            )
    
    async def process_batch_async(self, texts: List[str], model_type: str) -> List[Dict[str, Any]]:
        """Process multiple texts concurrently."""
        tasks = [
            self.process_text_async(text, model_type)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    def _sync_process(self, text: str, model_type: str) -> Dict[str, Any]:
        """Synchronous processing wrapper."""
        # Actual model processing logic
        pass
```

### Connection Pooling

```python
# src/utils/connection_pool.py
import asyncpg
import aioredis
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Manages database and cache connection pools."""
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.Redis] = None
    
    async def init_db_pool(self, database_url: str, min_size: int = 10, max_size: int = 50):
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=min_size,
                max_size=max_size,
                command_timeout=30,
                server_settings={
                    'jit': 'off',  # Disable JIT for faster connection
                    'application_name': 'nlp-pipeline'
                }
            )
            logger.info(f"Database pool initialized: {min_size}-{max_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def init_redis_pool(self, redis_url: str, max_connections: int = 50):
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = aioredis.from_url(
                redis_url,
                max_connections=max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                }
            )
            logger.info(f"Redis pool initialized: {max_connections} max connections")
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise
```

### Request Batching

```python
# src/api/batch_handler.py
import asyncio
from typing import List, Dict, Any, Callable
from collections import defaultdict
import time

class BatchHandler:
    """Intelligent request batching for improved throughput."""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.1,
                 processor: Callable = None):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.processor = processor
        self.pending_requests = defaultdict(list)
        self.batch_queues = defaultdict(asyncio.Queue)
        
    async def add_request(self, request_type: str, text: str) -> Dict[str, Any]:
        """Add request to batch queue."""
        future = asyncio.Future()
        
        # Add to pending requests
        self.pending_requests[request_type].append({
            'text': text,
            'future': future,
            'timestamp': time.time()
        })
        
        # Trigger batch processing if needed
        if len(self.pending_requests[request_type]) >= self.max_batch_size:
            asyncio.create_task(self._process_batch(request_type))
        else:
            # Schedule batch processing after max_wait_time
            asyncio.create_task(self._schedule_batch_processing(request_type))
        
        return await future
    
    async def _schedule_batch_processing(self, request_type: str):
        """Schedule batch processing after wait time."""
        await asyncio.sleep(self.max_wait_time)
        if self.pending_requests[request_type]:
            await self._process_batch(request_type)
    
    async def _process_batch(self, request_type: str):
        """Process accumulated batch."""
        if not self.pending_requests[request_type]:
            return
        
        # Extract batch
        batch = self.pending_requests[request_type][:self.max_batch_size]
        self.pending_requests[request_type] = self.pending_requests[request_type][self.max_batch_size:]
        
        try:
            # Process batch
            texts = [req['text'] for req in batch]
            results = await self.processor(texts, request_type)
            
            # Set results
            for req, result in zip(batch, results):
                req['future'].set_result(result)
                
        except Exception as e:
            # Set exception for all requests
            for req in batch:
                req['future'].set_exception(e)
```

## Model Optimization

### Model Quantization

```python
# src/models/optimization.py
import torch
from transformers import AutoTokenizer, AutoModel
import torch.quantization as quantization

class QuantizedModel:
    """Quantized model for faster inference."""
    
    def __init__(self, model_name: str, quantization_config: Dict[str, Any] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_quantized_model(quantization_config or {})
    
    def _load_quantized_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Load and quantize model."""
        model = AutoModel.from_pretrained(self.model_name)
        
        # Dynamic quantization (default)
        if config.get('type', 'dynamic') == 'dynamic':
            quantized_model = quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        # Static quantization
        elif config['type'] == 'static':
            model.eval()
            model.qconfig = quantization.get_default_qconfig('fbgemm')
            quantization.prepare(model, inplace=True)
            
            # Calibration would happen here with representative data
            # self._calibrate_model(model, calibration_data)
            
            quantized_model = quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized prediction with quantized model."""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Use mixed precision for faster inference
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
            
            return self._process_outputs(outputs)
```

### ONNX Runtime Optimization

```python
# src/models/onnx_optimizer.py
import onnxruntime as ort
import numpy as np
from typing import List, Dict, Any

class ONNXOptimizedModel:
    """ONNX Runtime optimized model for better performance."""
    
    def __init__(self, onnx_model_path: str, providers: List[str] = None):
        self.providers = providers or ['CPUExecutionProvider']
        
        # Configure session options for performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def predict_batch(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Batch prediction with ONNX runtime."""
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]  # Return logits
```

### Model Distillation

```python
# src/models/distillation.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DistilledSentimentModel(nn.Module):
    """Lightweight distilled model for sentiment analysis."""
    
    def __init__(self, 
                 teacher_model_name: str,
                 hidden_size: int = 384,
                 num_layers: int = 6):
        super().__init__()
        
        # Smaller transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=1536,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 3)  # positive, negative, neutral
        self.dropout = nn.Dropout(0.1)
        
        # Embedding layer
        self.embedding = nn.Embedding(30522, hidden_size)  # BERT vocab size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through distilled model."""
        # Embed tokens
        embeddings = self.embedding(input_ids)
        
        # Apply attention mask
        embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        # Pass through encoder
        encoded = self.encoder(embeddings)
        
        # Pool and classify
        pooled = encoded.mean(dim=1)  # Simple mean pooling
        pooled = self.dropout(pooled)
        
        return self.classifier(pooled)
```

## Caching Strategies

### Multi-Level Caching

```python
# src/cache/multi_level_cache.py
import asyncio
import hashlib
from typing import Any, Optional, Dict
import pickle
import time
from functools import wraps

class MultiLevelCache:
    """Multi-level caching system with L1 (memory) and L2 (Redis) cache."""
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l1_ttl: int = 300,
                 l2_ttl: int = 3600,
                 redis_client=None):
        
        # L1 Cache (In-memory)
        self.l1_cache: Dict[str, Dict[str, Any]] = {}
        self.l1_max_size = l1_max_size
        self.l1_ttl = l1_ttl
        
        # L2 Cache (Redis)
        self.redis_client = redis_client
        self.l2_ttl = l2_ttl
        
    def _generate_key(self, text: str, model_type: str) -> str:
        """Generate cache key from text and model type."""
        content = f"{model_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, text: str, model_type: str) -> Optional[Any]:
        """Get value from cache (L1 then L2)."""
        key = self._generate_key(text, model_type)
        
        # Try L1 cache first
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if time.time() - entry['timestamp'] < self.l1_ttl:
                return entry['value']
            else:
                # Expired, remove from L1
                del self.l1_cache[key]
        
        # Try L2 cache (Redis)
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"cache:{key}")
                if cached_data:
                    value = pickle.loads(cached_data)
                    # Promote to L1 cache
                    self._set_l1(key, value)
                    return value
            except Exception:
                pass  # Redis error, continue without cache
        
        return None
    
    async def set(self, text: str, model_type: str, value: Any) -> None:
        """Set value in both cache levels."""
        key = self._generate_key(text, model_type)
        
        # Set in L1 cache
        self._set_l1(key, value)
        
        # Set in L2 cache (Redis)
        if self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                await self.redis_client.setex(
                    f"cache:{key}",
                    self.l2_ttl,
                    serialized_value
                )
            except Exception:
                pass  # Redis error, continue with L1 only
    
    def _set_l1(self, key: str, value: Any) -> None:
        """Set value in L1 cache with LRU eviction."""
        # LRU eviction if cache is full
        if len(self.l1_cache) >= self.l1_max_size:
            oldest_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k]['timestamp']
            )
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

# Cache decorator
def cached_prediction(cache_instance: MultiLevelCache):
    """Decorator for caching model predictions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, text: str, model_type: str = None, **kwargs):
            # Try cache first
            cached_result = await cache_instance.get(text, model_type or func.__name__)
            if cached_result is not None:
                return cached_result
            
            # Call original function
            result = await func(self, text, model_type, **kwargs)
            
            # Cache result
            await cache_instance.set(text, model_type or func.__name__, result)
            
            return result
        return wrapper
    return decorator
```

### Precomputation Cache

```python
# src/cache/precompute.py
import asyncio
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PrecomputeCache:
    """Precompute and cache common predictions."""
    
    def __init__(self, model_manager, cache_manager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.common_texts = []
    
    async def precompute_common_texts(self, texts: List[str]) -> None:
        """Precompute predictions for common texts."""
        logger.info(f"Precomputing predictions for {len(texts)} texts")
        
        # Process in batches to avoid overwhelming the system
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Precompute sentiment analysis
            await self._precompute_batch(batch, 'sentiment')
            
            # Precompute entity extraction
            await self._precompute_batch(batch, 'entity')
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        logger.info("Precomputation completed")
    
    async def _precompute_batch(self, texts: List[str], model_type: str) -> None:
        """Precompute predictions for a batch of texts."""
        try:
            # Get model predictions
            if model_type == 'sentiment':
                results = await self.model_manager.predict_sentiment_batch(texts)
            elif model_type == 'entity':
                results = await self.model_manager.predict_entity_batch(texts)
            else:
                return
            
            # Cache results
            for text, result in zip(texts, results):
                await self.cache_manager.set(text, model_type, result)
                
        except Exception as e:
            logger.error(f"Precomputation failed for {model_type}: {e}")
    
    async def warm_cache_from_logs(self, log_file_path: str, top_n: int = 1000) -> None:
        """Warm cache using most frequent texts from logs."""
        # Extract and count frequent texts from logs
        frequent_texts = self._extract_frequent_texts(log_file_path, top_n)
        
        # Precompute for frequent texts
        await self.precompute_common_texts(frequent_texts)
```

## Database Optimization

### Query Optimization

```sql
-- Database schema optimizations

-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_predictions_text_hash ON predictions USING hash(text_hash);
CREATE INDEX CONCURRENTLY idx_predictions_created_at ON predictions (created_at);
CREATE INDEX CONCURRENTLY idx_predictions_model_type ON predictions (model_type);

-- Composite index for common filter combinations
CREATE INDEX CONCURRENTLY idx_predictions_composite 
ON predictions (model_type, created_at) 
WHERE confidence > 0.8;

-- Partial index for high-confidence predictions
CREATE INDEX CONCURRENTLY idx_high_confidence_predictions 
ON predictions (text_hash, model_type) 
WHERE confidence > 0.9;

-- Optimize text search
CREATE INDEX CONCURRENTLY idx_entities_text_gin 
ON entities USING gin(to_tsvector('english', entity_text));
```

```python
# src/database/optimized_queries.py
import asyncpg
from typing import List, Dict, Any, Optional

class OptimizedQueries:
    """Optimized database queries for better performance."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
    
    async def get_cached_prediction(self, text_hash: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction with optimized query."""
        query = """
            SELECT prediction_data, confidence, created_at
            FROM predictions 
            WHERE text_hash = $1 AND model_type = $2
            AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 1
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, text_hash, model_type)
            return dict(row) if row else None
    
    async def bulk_insert_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Bulk insert predictions for better performance."""
        query = """
            INSERT INTO predictions (text_hash, model_type, prediction_data, confidence, created_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (text_hash, model_type) DO UPDATE SET
                prediction_data = EXCLUDED.prediction_data,
                confidence = EXCLUDED.confidence,
                created_at = EXCLUDED.created_at
        """
        
        # Prepare data for bulk insert
        data = [
            (
                pred['text_hash'],
                pred['model_type'],
                pred['prediction_data'],
                pred['confidence'],
                pred['created_at']
            )
            for pred in predictions
        ]
        
        async with self.db_pool.acquire() as conn:
            await conn.executemany(query, data)
    
    async def get_frequent_texts(self, limit: int = 1000) -> List[str]:
        """Get most frequently processed texts."""
        query = """
            SELECT text_hash, COUNT(*) as frequency
            FROM predictions 
            WHERE created_at > NOW() - INTERVAL '7 days'
            GROUP BY text_hash
            ORDER BY frequency DESC
            LIMIT $1
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, limit)
            return [row['text_hash'] for row in rows]
```

### Connection Pool Optimization

```python
# src/database/pool_config.py
import asyncpg
import logging

logger = logging.getLogger(__name__)

async def create_optimized_pool(database_url: str, 
                              min_size: int = 10,
                              max_size: int = 50) -> asyncpg.Pool:
    """Create optimized database connection pool."""
    
    pool = await asyncpg.create_pool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        
        # Connection timeout settings
        command_timeout=30,
        server_settings={
            # Disable JIT for faster connection establishment
            'jit': 'off',
            
            # Optimize for OLTP workload
            'shared_preload_libraries': 'pg_stat_statements',
            'track_activity_query_size': '2048',
            'track_functions': 'all',
            
            # Memory settings
            'work_mem': '64MB',
            'maintenance_work_mem': '256MB',
            
            # Connection settings
            'application_name': 'nlp-pipeline',
            'tcp_keepalives_idle': '300',
            'tcp_keepalives_interval': '30',
            'tcp_keepalives_count': '3',
        },
        
        # Connection lifecycle callbacks
        init=_init_connection,
        setup=_setup_connection,
    )
    
    logger.info(f"Database pool created: {min_size}-{max_size} connections")
    return pool

async def _init_connection(conn: asyncpg.Connection) -> None:
    """Initialize connection with optimizations."""
    # Set up prepared statements for common queries
    await conn.execute("PREPARE get_prediction AS SELECT * FROM predictions WHERE text_hash = $1")
    
    # Set connection-level settings
    await conn.execute("SET statement_timeout = '30s'")
    await conn.execute("SET lock_timeout = '10s'")

async def _setup_connection(conn: asyncpg.Connection) -> None:
    """Setup connection with custom types."""
    # Register custom types if needed
    pass
```

## Network Optimization

### HTTP/2 and Connection Reuse

```python
# src/api/optimized_client.py
import aiohttp
import asyncio
from typing import Dict, Any, List

class OptimizedHTTPClient:
    """Optimized HTTP client for external API calls."""
    
    def __init__(self, 
                 connector_limit: int = 100,
                 connector_limit_per_host: int = 30,
                 timeout: float = 30.0):
        
        # Configure connector for connection pooling
        connector = aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=connector_limit_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300,
        )
        
        # Configure timeout
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers={
                'User-Agent': 'NLP-Pipeline/1.0',
                'Connection': 'keep-alive',
            }
        )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

### Response Compression

```python
# src/api/compression.py
import gzip
import json
from typing import Any, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse

class CompressedJSONResponse(JSONResponse):
    """JSON response with compression for large payloads."""
    
    def __init__(self, content: Any, **kwargs):
        super().__init__(content, **kwargs)
        
        # Compress response if it's large enough
        json_str = json.dumps(content)
        if len(json_str) > 1024:  # Compress if > 1KB
            compressed = gzip.compress(json_str.encode())
            if len(compressed) < len(json_str) * 0.8:  # Only if compression saves 20%+
                self.body = compressed
                self.headers['content-encoding'] = 'gzip'
                self.headers['vary'] = 'Accept-Encoding'
```

## Memory Management

### Memory Pool for Models

```python
# src/models/memory_pool.py
import torch
from typing import Dict, Any, Optional
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

class ModelMemoryPool:
    """Memory pool for efficient model management."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.memory_usage: Dict[str, int] = {}
    
    def can_load_model(self, estimated_size_mb: int) -> bool:
        """Check if model can be loaded within memory limits."""
        estimated_size_bytes = estimated_size_mb * 1024**2
        current_usage = sum(self.memory_usage.values())
        
        return (current_usage + estimated_size_bytes) <= self.max_memory_bytes
    
    def load_model(self, model_key: str, model_object: Any) -> bool:
        """Load model into memory pool."""
        # Estimate model size
        model_size = self._estimate_model_size(model_object)
        
        # Check if we can load the model
        if not self.can_load_model(model_size // (1024**2)):
            # Try to free memory by unloading least recently used models
            self._free_memory_lru(model_size)
        
        # Load model if we have space
        if self.can_load_model(model_size // (1024**2)):
            self.loaded_models[model_key] = {
                'model': model_object,
                'last_used': time.time(),
                'load_time': time.time()
            }
            self.memory_usage[model_key] = model_size
            logger.info(f"Model {model_key} loaded, using {model_size / (1024**2):.1f} MB")
            return True
        
        return False
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model memory usage."""
        if hasattr(model, 'parameters'):
            # PyTorch model
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size
        else:
            # Generic object
            return sys.getsizeof(model)
    
    def _free_memory_lru(self, required_bytes: int) -> None:
        """Free memory by unloading least recently used models."""
        # Sort models by last used time
        sorted_models = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1]['last_used']
        )
        
        freed_bytes = 0
        for model_key, model_info in sorted_models:
            if freed_bytes >= required_bytes:
                break
            
            # Unload model
            self.unload_model(model_key)
            freed_bytes += self.memory_usage.get(model_key, 0)
    
    def unload_model(self, model_key: str) -> None:
        """Unload model from memory."""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            if model_key in self.memory_usage:
                freed_mb = self.memory_usage[model_key] / (1024**2)
                del self.memory_usage[model_key]
                logger.info(f"Model {model_key} unloaded, freed {freed_mb:.1f} MB")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### Gradient Accumulation

```python
# src/models/efficient_training.py
import torch
from typing import List, Dict, Any

class EfficientTrainer:
    """Memory-efficient training with gradient accumulation."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    
    def train_batch(self, batch_data: List[Dict[str, Any]]) -> float:
        """Train with gradient accumulation and mixed precision."""
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # Process in sub-batches for gradient accumulation
        sub_batch_size = len(batch_data) // self.accumulation_steps
        
        for step in range(self.accumulation_steps):
            start_idx = step * sub_batch_size
            end_idx = start_idx + sub_batch_size
            sub_batch = batch_data[start_idx:end_idx]
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(sub_batch)
                loss = loss / self.accumulation_steps  # Scale loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss
```

## CPU Optimization

### Multiprocessing for CPU-bound Tasks

```python
# src/processing/cpu_optimizer.py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Callable
import numpy as np

class CPUOptimizer:
    """CPU optimization utilities for parallel processing."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def parallel_process(self, 
                        func: Callable,
                        data: List[Any],
                        chunk_size: int = None) -> List[Any]:
        """Process data in parallel across CPU cores."""
        if chunk_size is None:
            chunk_size = max(1, len(data) // self.max_workers)
        
        # Split data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        futures = [self.executor.submit(func, chunk) for chunk in chunks]
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results
    
    def vectorized_text_processing(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Vectorized text preprocessing using NumPy."""
        # Convert to numpy array for vectorized operations
        text_array = np.array(texts, dtype=object)
        
        # Vectorized operations
        lengths = np.vectorize(len)(text_array)
        word_counts = np.vectorize(lambda x: len(x.split()))(text_array)
        
        # Return structured results
        return [
            {
                'text': text,
                'length': length,
                'word_count': word_count
            }
            for text, length, word_count in zip(texts, lengths, word_counts)
        ]
```

### SIMD Optimizations

```python
# src/processing/simd_optimizer.py
import numpy as np
from numba import jit, vectorize
import math

@jit(nopython=True)
def fast_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """JIT-compiled cosine similarity calculation."""
    dot_product = np.dot(a, b)
    norm_a = math.sqrt(np.dot(a, a))
    norm_b = math.sqrt(np.dot(b, b))
    return dot_product / (norm_a * norm_b)

@vectorize(['float64(float64)'], nopython=True)
def fast_sigmoid(x):
    """Vectorized sigmoid function."""
    if x > 0:
        exp_neg_x = math.exp(-x)
        return 1.0 / (1.0 + exp_neg_x)
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

class SIMDProcessor:
    """SIMD-optimized processing functions."""
    
    @staticmethod
    def batch_similarity(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Compute similarities using SIMD operations."""
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query / np.linalg.norm(query)
        
        # Vectorized dot product
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    @staticmethod
    def batch_softmax(logits: np.ndarray) -> np.ndarray:
        """Vectorized softmax computation."""
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        
        # Vectorized exponential
        exp_logits = np.exp(shifted_logits)
        
        # Normalize
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

## Monitoring and Profiling

### Performance Profiler

```python
# src/monitoring/profiler.py
import cProfile
import pstats
import time
import psutil
import tracemalloc
from typing import Dict, Any, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Comprehensive performance profiler for the NLP pipeline."""
    
    def __init__(self):
        self.profiles: Dict[str, Any] = {}
        self.memory_snapshots: Dict[str, Any] = {}
        
    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                
                # Start profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                # Start memory tracking
                tracemalloc.start()
                start_memory = psutil.Process().memory_info().rss
                start_time = time.time()
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Stop profiling
                    profiler.disable()
                    
                    # Collect metrics
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    # Store profile data
                    self.profiles[name] = {
                        'execution_time': end_time - start_time,
                        'memory_usage_mb': (end_memory - start_memory) / (1024**2),
                        'peak_memory_mb': peak / (1024**2),
                        'profiler': profiler
                    }
                    
                    logger.info(f"{name} - Time: {end_time - start_time:.3f}s, "
                              f"Memory: {(end_memory - start_memory) / (1024**2):.1f}MB")
                    
                    return result
                    
                except Exception as e:
                    profiler.disable()
                    tracemalloc.stop()
                    raise e
                    
            return wrapper
        return decorator
    
    def get_profile_stats(self, func_name: str, sort_by: str = 'cumulative') -> str:
        """Get detailed profile statistics."""
        if func_name not in self.profiles:
            return f"No profile data for {func_name}"
        
        profiler = self.profiles[func_name]['profiler']
        stats = pstats.Stats(profiler)
        stats.sort_stats(sort_by)
        
        # Capture stats output
        import io
        output = io.StringIO()
        stats.print_stats(output_stream=output)
        
        return output.getvalue()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for name, data in self.profiles.items():
            summary[name] = {
                'execution_time': data['execution_time'],
                'memory_usage_mb': data['memory_usage_mb'],
                'peak_memory_mb': data['peak_memory_mb']
            }
        return summary
```

### Real-time Metrics

```python
# src/monitoring/real_time_metrics.py
import time
import asyncio
from collections import deque, defaultdict
from typing import Dict, Any, Deque
import statistics

class RealTimeMetrics:
    """Real-time performance metrics collection."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
    
    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record operation latency."""
        self.metrics[f"{operation}_latency"].append(latency_ms)
        self.counters[f"{operation}_count"] += 1
    
    def record_throughput(self, operation: str, count: int = 1) -> None:
        """Record throughput metric."""
        self.counters[f"{operation}_throughput"] += count
        self.metrics[f"{operation}_throughput_ts"].append(time.time())
    
    def get_percentiles(self, metric_name: str) -> Dict[str, float]:
        """Get percentile statistics for a metric."""
        if metric_name not in self.metrics:
            return {}
        
        values = list(self.metrics[metric_name])
        if not values:
            return {}
        
        return {
            'p50': statistics.median(values),
            'p90': statistics.quantiles(values, n=10)[8] if len(values) >= 10 else max(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            'mean': statistics.mean(values),
            'max': max(values),
            'min': min(values)
        }
    
    def get_throughput(self, operation: str, time_window_seconds: int = 60) -> float:
        """Get throughput for an operation."""
        metric_name = f"{operation}_throughput_ts"
        if metric_name not in self.metrics:
            return 0.0
        
        current_time = time.time()
        window_start = current_time - time_window_seconds
        
        # Count events in time window
        count = sum(1 for ts in self.metrics[metric_name] if ts >= window_start)
        
        return count / time_window_seconds
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""
        dashboard = {
            'uptime_seconds': time.time() - self.start_time,
            'latencies': {},
            'throughputs': {},
            'counters': dict(self.counters)
        }
        
        # Collect latency statistics
        for metric_name in self.metrics:
            if metric_name.endswith('_latency'):
                operation = metric_name.replace('_latency', '')
                dashboard['latencies'][operation] = self.get_percentiles(metric_name)
        
        # Collect throughput statistics
        for metric_name in self.metrics:
            if metric_name.endswith('_throughput_ts'):
                operation = metric_name.replace('_throughput_ts', '')
                dashboard['throughputs'][operation] = {
                    '1min': self.get_throughput(operation, 60),
                    '5min': self.get_throughput(operation, 300),
                    '15min': self.get_throughput(operation, 900)
                }
        
        return dashboard
```

## Load Testing

### Load Test Configuration

```python
# scripts/load_test.py
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any
import json

class LoadTester:
    """Load testing utility for the NLP API."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.results: List[Dict[str, Any]] = []
    
    async def test_sentiment_analysis(self, 
                                    concurrent_users: int = 10,
                                    requests_per_user: int = 100,
                                    test_texts: List[str] = None) -> Dict[str, Any]:
        """Load test sentiment analysis endpoint."""
        
        if not test_texts:
            test_texts = [
                "I love this product!",
                "This is terrible.",
                "It's okay, nothing special.",
                "Amazing experience!",
                "Worst service ever."
            ]
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        async with aiohttp.ClientSession(headers=headers) as session:
            # Create tasks for concurrent users
            tasks = []
            for user_id in range(concurrent_users):
                task = self._user_load_test(
                    session, 
                    user_id, 
                    requests_per_user, 
                    test_texts
                )
                tasks.append(task)
            
            # Run load test
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Aggregate results
            return self._aggregate_results(results, start_time, end_time)
    
    async def _user_load_test(self, 
                            session: aiohttp.ClientSession,
                            user_id: int,
                            requests: int,
                            test_texts: List[str]) -> List[Dict[str, Any]]:
        """Simulate load for a single user."""
        user_results = []
        
        for i in range(requests):
            text = test_texts[i % len(test_texts)]
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/sentiment/analyze",
                    json={"text": text}
                ) as response:
                    end_time = time.time()
                    
                    result = {
                        'user_id': user_id,
                        'request_id': i,
                        'latency_ms': (end_time - start_time) * 1000,
                        'status_code': response.status,
                        'success': response.status == 200
                    }
                    
                    if response.status == 200:
                        data = await response.json()
                        result['response_data'] = data
                    
                    user_results.append(result)
                    
            except Exception as e:
                end_time = time.time()
                user_results.append({
                    'user_id': user_id,
                    'request_id': i,
                    'latency_ms': (end_time - start_time) * 1000,
                    'status_code': 0,
                    'success': False,
                    'error': str(e)
                })
        
        return user_results
    
    def _aggregate_results(self, 
                          results: List[List[Dict[str, Any]]],
                          start_time: float,
                          end_time: float) -> Dict[str, Any]:
        """Aggregate load test results."""
        all_results = [result for user_results in results for result in user_results]
        
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        latencies = [r['latency_ms'] for r in successful_requests]
        
        if latencies:
            latency_stats = {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p90': statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else max(latencies),
                'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
                'min': min(latencies),
                'max': max(latencies)
            }
        else:
            latency_stats = {}
        
        total_duration = end_time - start_time
        total_requests = len(all_results)
        successful_requests_count = len(successful_requests)
        
        return {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests_count,
                'failed_requests': len(failed_requests),
                'success_rate': successful_requests_count / total_requests if total_requests > 0 else 0,
                'total_duration_seconds': total_duration,
                'requests_per_second': total_requests / total_duration,
                'successful_requests_per_second': successful_requests_count / total_duration
            },
            'latency_stats': latency_stats,
            'error_distribution': self._get_error_distribution(failed_requests),
            'detailed_results': all_results
        }
    
    def _get_error_distribution(self, failed_requests: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of error types."""
        error_counts = {}
        for request in failed_requests:
            error_type = request.get('error', f"HTTP {request['status_code']}")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

# Usage example
async def run_load_test():
    tester = LoadTester("http://localhost:8000/v1")
    results = await tester.test_sentiment_analysis(
        concurrent_users=50,
        requests_per_user=200
    )
    
    print(json.dumps(results['summary'], indent=2))
    print(f"Latency P95: {results['latency_stats']['p95']:.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_load_test())
```

## Best Practices

### 1. Batch Processing

```python
# Always prefer batch processing over individual requests
# Bad
results = []
for text in texts:
    result = model.predict(text)
    results.append(result)

# Good
results = model.predict_batch(texts, batch_size=32)
```

### 2. Async/Await Pattern

```python
# Use async/await for I/O bound operations
async def process_request(text: str) -> Dict[str, Any]:
    # Non-blocking database query
    cached_result = await cache.get(text)
    if cached_result:
        return cached_result
    
    # Non-blocking model prediction
    result = await model.predict_async(text)
    
    # Non-blocking cache storage
    await cache.set(text, result)
    
    return result
```

### 3. Memory Management

```python
# Clear intermediate variables in large loops
for batch in large_dataset:
    processed_batch = expensive_operation(batch)
    save_results(processed_batch)
    
    # Clear memory
    del processed_batch
    gc.collect()
```

### 4. Connection Pooling

```python
# Use connection pools for database and external services
db_pool = await asyncpg.create_pool(database_url, min_size=10, max_size=50)
redis_pool = aioredis.from_url(redis_url, max_connections=50)
```

### 5. Caching Strategy

```python
# Implement multi-level caching
@cached_prediction(cache_instance=multi_level_cache)
async def predict_sentiment(text: str) -> Dict[str, Any]:
    # This will be cached automatically
    return await model.predict(text)
```

### 6. Error Handling

```python
# Implement circuit breaker pattern for external dependencies
from circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, recovery_timeout=30)
async def external_api_call():
    # This will fail fast if external service is down
    pass
```

### 7. Monitoring

```python
# Add performance monitoring to critical functions
@performance_profiler.profile_function("sentiment_analysis")
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    with metrics.timer("sentiment_analysis"):
        return await model.predict(text)
```

This performance guide provides comprehensive strategies for optimizing every aspect of the NLP Pipeline, from model inference to database queries and network communication. Regular monitoring and profiling are essential to maintain optimal performance as the system scales.