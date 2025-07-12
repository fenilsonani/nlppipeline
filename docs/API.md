# API Documentation

## Overview

The NLP Pipeline provides a comprehensive REST API for text processing, sentiment analysis, and entity extraction. The API is built with FastAPI and follows OpenAPI specifications for easy integration and testing.

## Base URL

```
Production: https://api.nlp-pipeline.com/v1
Staging: https://staging-api.nlp-pipeline.com/v1
Development: http://localhost:8000/v1
```

## Authentication

### API Key Authentication

```http
Authorization: Bearer your-api-key-here
```

### JWT Token Authentication

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Rate Limiting

- **Free Tier**: 100 requests/hour
- **Basic Plan**: 1,000 requests/hour  
- **Pro Plan**: 10,000 requests/hour
- **Enterprise**: Unlimited with SLA

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Response Format

All API responses follow a consistent format:

### Success Response

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time_ms": 150,
    "version": "1.2.0"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text input cannot be empty",
    "details": {
      "field": "text",
      "constraint": "min_length"
    }
  },
  "metadata": {
    "request_id": "req_def456",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.2.0"
  }
}
```

## Endpoints

### Health Check

#### GET /health

Check the health status of the API and its dependencies.

**Response**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "components": {
      "api": {
        "status": "healthy",
        "response_time_ms": 2
      },
      "database": {
        "status": "healthy",
        "response_time_ms": 15
      },
      "models": {
        "status": "healthy",
        "loaded_models": ["sentiment", "entity"],
        "memory_usage_mb": 2048
      },
      "cache": {
        "status": "healthy",
        "hit_rate": 0.85
      }
    }
  }
}
```

### Sentiment Analysis

#### POST /sentiment/analyze

Analyze sentiment of input text(s).

**Request Body**

```json
{
  "text": "I love this product! It's amazing!",
  "options": {
    "return_scores": true,
    "language": "en"
  }
}
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string \| array | Yes | Single text or array of texts to analyze |
| options.return_scores | boolean | No | Return confidence scores for all sentiments (default: false) |
| options.language | string | No | Language code (default: auto-detect) |

**Response**

```json
{
  "success": true,
  "data": {
    "text": "I love this product! It's amazing!",
    "sentiment": "positive",
    "confidence": 0.95,
    "scores": {
      "positive": 0.95,
      "negative": 0.03,
      "neutral": 0.02
    },
    "language": "en"
  }
}
```

#### POST /sentiment/batch

Analyze sentiment for multiple texts in batch.

**Request Body**

```json
{
  "texts": [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
  ],
  "options": {
    "return_scores": true,
    "batch_size": 32
  }
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "I love this product!",
        "sentiment": "positive",
        "confidence": 0.95,
        "scores": {
          "positive": 0.95,
          "negative": 0.03,
          "neutral": 0.02
        }
      },
      {
        "text": "This is terrible.",
        "sentiment": "negative",
        "confidence": 0.89,
        "scores": {
          "positive": 0.05,
          "negative": 0.89,
          "neutral": 0.06
        }
      }
    ],
    "summary": {
      "total_texts": 3,
      "sentiment_distribution": {
        "positive": 1,
        "negative": 1,
        "neutral": 1
      },
      "average_confidence": 0.88
    }
  }
}
```

### Entity Extraction

#### POST /entities/extract

Extract named entities from input text.

**Request Body**

```json
{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
  "options": {
    "entity_types": ["Person", "Organization", "Location"],
    "include_confidence": true,
    "merge_entities": true
  }
}
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | Text to analyze for entities |
| options.entity_types | array | No | Filter specific entity types |
| options.include_confidence | boolean | No | Include confidence scores (default: false) |
| options.merge_entities | boolean | No | Merge overlapping entities (default: true) |

**Response**

```json
{
  "success": true,
  "data": {
    "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "entities": [
      {
        "text": "Apple Inc.",
        "type": "Organization",
        "start": 0,
        "end": 10,
        "confidence": 0.99
      },
      {
        "text": "Steve Jobs",
        "type": "Person",
        "start": 26,
        "end": 36,
        "confidence": 0.98
      },
      {
        "text": "Cupertino",
        "type": "Location",
        "start": 40,
        "end": 49,
        "confidence": 0.95
      },
      {
        "text": "California",
        "type": "Location",
        "start": 51,
        "end": 61,
        "confidence": 0.97
      }
    ],
    "total_entities": 4,
    "entity_counts": {
      "Organization": 1,
      "Person": 1,
      "Location": 2
    }
  }
}
```

#### POST /entities/batch

Extract entities from multiple texts in batch.

**Request Body**

```json
{
  "texts": [
    "Apple Inc. was founded by Steve Jobs.",
    "Microsoft announced a partnership with OpenAI."
  ],
  "options": {
    "entity_types": ["Organization", "Person"],
    "batch_size": 16
  }
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "text": "Apple Inc. was founded by Steve Jobs.",
        "entities": [
          {
            "text": "Apple Inc.",
            "type": "Organization",
            "start": 0,
            "end": 10
          },
          {
            "text": "Steve Jobs",
            "type": "Person",
            "start": 26,
            "end": 36
          }
        ],
        "total_entities": 2
      }
    ],
    "summary": {
      "total_texts": 2,
      "total_entities": 4,
      "entity_distribution": {
        "Organization": 2,
        "Person": 1
      }
    }
  }
}
```

### Combined Analysis

#### POST /analyze

Perform both sentiment analysis and entity extraction in a single request.

**Request Body**

```json
{
  "text": "Apple's new iPhone launch was absolutely amazing!",
  "analyses": ["sentiment", "entities"],
  "options": {
    "sentiment": {
      "return_scores": true
    },
    "entities": {
      "entity_types": ["Organization", "Product"],
      "include_confidence": true
    }
  }
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "text": "Apple's new iPhone launch was absolutely amazing!",
    "sentiment": {
      "sentiment": "positive",
      "confidence": 0.92,
      "scores": {
        "positive": 0.92,
        "negative": 0.04,
        "neutral": 0.04
      }
    },
    "entities": {
      "entities": [
        {
          "text": "Apple",
          "type": "Organization",
          "start": 0,
          "end": 5,
          "confidence": 0.99
        },
        {
          "text": "iPhone",
          "type": "Product",
          "start": 12,
          "end": 18,
          "confidence": 0.95
        }
      ],
      "total_entities": 2
    }
  }
}
```

### Model Management

#### GET /models

List available models and their status.

**Response**

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "sentiment",
        "version": "2.1.0",
        "type": "sentiment_analysis",
        "status": "loaded",
        "memory_usage_mb": 512,
        "load_time": "2024-01-15T10:00:00Z",
        "last_used": "2024-01-15T10:29:45Z",
        "requests_served": 1543
      },
      {
        "name": "entity",
        "version": "1.8.0",
        "type": "named_entity_recognition",
        "status": "loaded",
        "memory_usage_mb": 768,
        "load_time": "2024-01-15T10:00:15Z",
        "last_used": "2024-01-15T10:29:50Z",
        "requests_served": 892
      }
    ],
    "total_models": 2,
    "total_memory_mb": 1280
  }
}
```

#### POST /models/{model_name}/load

Load a specific model into memory.

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model_name | string | Yes | Name of the model to load |

**Request Body**

```json
{
  "version": "2.1.0",
  "options": {
    "device": "cuda",
    "precision": "fp16"
  }
}
```

**Response**

```json
{
  "success": true,
  "data": {
    "model_name": "sentiment",
    "version": "2.1.0",
    "status": "loaded",
    "load_time_ms": 2500,
    "memory_usage_mb": 512
  }
}
```

#### DELETE /models/{model_name}

Unload a model from memory.

**Response**

```json
{
  "success": true,
  "data": {
    "model_name": "sentiment",
    "status": "unloaded",
    "memory_freed_mb": 512
  }
}
```

### Monitoring

#### GET /metrics

Get performance metrics for the API.

**Response**

```json
{
  "success": true,
  "data": {
    "api_metrics": {
      "total_requests": 15432,
      "requests_per_minute": 127,
      "average_response_time_ms": 145,
      "error_rate": 0.02
    },
    "model_metrics": {
      "sentiment": {
        "total_predictions": 8765,
        "average_prediction_time_ms": 23,
        "cache_hit_rate": 0.65
      },
      "entity": {
        "total_predictions": 5234,
        "average_prediction_time_ms": 35,
        "cache_hit_rate": 0.58
      }
    },
    "system_metrics": {
      "cpu_usage_percent": 45,
      "memory_usage_percent": 68,
      "disk_usage_percent": 23
    }
  }
}
```

### Streaming API

#### WebSocket /ws/stream

Real-time streaming analysis via WebSocket connection.

**Connection**

```javascript
const ws = new WebSocket('wss://api.nlp-pipeline.com/v1/ws/stream?token=your-api-key');
```

**Send Message**

```json
{
  "type": "analyze",
  "data": {
    "text": "This is a test message",
    "analyses": ["sentiment", "entities"]
  },
  "request_id": "req_123"
}
```

**Receive Message**

```json
{
  "type": "result",
  "request_id": "req_123",
  "data": {
    "text": "This is a test message",
    "sentiment": {
      "sentiment": "neutral",
      "confidence": 0.78
    },
    "entities": {
      "entities": [],
      "total_entities": 0
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## SDKs and Libraries

### Python SDK

```python
from nlp_pipeline import NLPClient

client = NLPClient(api_key="your-api-key")

# Sentiment analysis
result = client.analyze_sentiment("I love this product!")
print(result.sentiment)  # 'positive'

# Entity extraction
entities = client.extract_entities("Apple Inc. was founded by Steve Jobs.")
for entity in entities:
    print(f"{entity.text} - {entity.type}")
```

### JavaScript SDK

```javascript
import { NLPClient } from '@nlp-pipeline/client';

const client = new NLPClient({ apiKey: 'your-api-key' });

// Sentiment analysis
const sentiment = await client.analyzeSentiment('I love this product!');
console.log(sentiment.sentiment); // 'positive'

// Entity extraction
const entities = await client.extractEntities('Apple Inc. was founded by Steve Jobs.');
entities.forEach(entity => {
  console.log(`${entity.text} - ${entity.type}`);
});
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_INPUT | 400 | Request body validation failed |
| MISSING_TEXT | 400 | Text field is required |
| TEXT_TOO_LONG | 400 | Text exceeds maximum length |
| INVALID_LANGUAGE | 400 | Unsupported language code |
| UNAUTHORIZED | 401 | Invalid or missing API key |
| FORBIDDEN | 403 | Insufficient permissions |
| RATE_LIMITED | 429 | Rate limit exceeded |
| MODEL_NOT_FOUND | 404 | Requested model not available |
| MODEL_LOAD_ERROR | 500 | Failed to load model |
| PROCESSING_ERROR | 500 | Error during text processing |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## Webhooks

Configure webhooks to receive notifications about processing results, model updates, and system events.

### Webhook Configuration

```json
{
  "url": "https://your-app.com/webhooks/nlp",
  "events": ["processing.completed", "model.updated", "alert.triggered"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload

```json
{
  "event": "processing.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "batch_id": "batch_abc123",
    "total_items": 1000,
    "completed_items": 1000,
    "failed_items": 0,
    "results_url": "https://api.nlp-pipeline.com/v1/batches/batch_abc123/results"
  },
  "signature": "sha256=..."
}
```

## Testing

### Interactive API Documentation

Visit the interactive API documentation at:
- Production: https://api.nlp-pipeline.com/docs
- Staging: https://staging-api.nlp-pipeline.com/docs

### cURL Examples

#### Sentiment Analysis

```bash
curl -X POST "https://api.nlp-pipeline.com/v1/sentiment/analyze" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product!",
    "options": {
      "return_scores": true
    }
  }'
```

#### Entity Extraction

```bash
curl -X POST "https://api.nlp-pipeline.com/v1/entities/extract" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc. was founded by Steve Jobs.",
    "options": {
      "entity_types": ["Organization", "Person"],
      "include_confidence": true
    }
  }'
```

## Best Practices

### 1. Batch Processing

For multiple texts, use batch endpoints instead of making multiple single requests:

```python
# Good - Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
results = client.analyze_sentiment_batch(texts)

# Bad - Multiple single requests
results = []
for text in texts:
    result = client.analyze_sentiment(text)
    results.append(result)
```

### 2. Caching

Implement client-side caching for repeated analyses:

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_sentiment_analysis(text_hash):
    return client.analyze_sentiment(text)

def analyze_with_cache(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return cached_sentiment_analysis(text_hash)
```

### 3. Error Handling

Always implement proper error handling:

```python
try:
    result = client.analyze_sentiment(text)
except APIError as e:
    if e.code == "RATE_LIMITED":
        # Implement backoff and retry
        time.sleep(60)
        result = client.analyze_sentiment(text)
    else:
        # Log error and handle gracefully
        logger.error(f"API Error: {e.message}")
        result = None
```

### 4. Rate Limiting

Respect rate limits and implement exponential backoff:

```python
import time
import random

def make_request_with_backoff(func, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
```

This API documentation provides comprehensive information for integrating with the NLP Pipeline service, including all endpoints, request/response formats, error handling, and best practices for optimal performance.