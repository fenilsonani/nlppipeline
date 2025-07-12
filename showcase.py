#!/usr/bin/env python3
"""
NLP Pipeline Feature Showcase

This script demonstrates all the key features of the Enterprise-Grade NLP Pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import NLPPipeline
from src.preprocessing import TextCleaner, BERTTokenizer
from src.models import SentimentAnalyzer, EntityExtractor
from src.postprocessing import Visualizer, ResultAggregator
from src.monitoring import metrics_collector


async def showcase_features():
    """Showcase all NLP Pipeline features."""
    print("🚀 NLP Pipeline Feature Showcase")
    print("="*80)
    
    # 1. Text Preprocessing
    print("\n1️⃣ TEXT PREPROCESSING")
    print("-"*40)
    cleaner = TextCleaner()
    messy_text = """
    Check out our AMAZING new product!!! 🎉🎉🎉
    Visit https://example.com for more details...
    Contact: support@example.com #BestProduct #Innovation
    """
    clean_text = cleaner.clean(messy_text)
    print(f"Original text: {messy_text.strip()}")
    print(f"Cleaned text: {clean_text}")
    
    # 2. Tokenization
    print("\n2️⃣ TOKENIZATION")
    print("-"*40)
    tokenizer = BERTTokenizer(model_name="distilbert-base-uncased")
    tokens = tokenizer.tokenize("Natural Language Processing is fascinating!")
    print(f"Tokens: {tokens.tokens[:10]}...")
    print(f"Token IDs: {tokens.token_ids[:10]}...")
    
    # 3. Sentiment Analysis
    print("\n3️⃣ SENTIMENT ANALYSIS")
    print("-"*40)
    analyzer = SentimentAnalyzer(model_name="distilbert-base-uncased-finetuned-sst-2-english")
    analyzer.load_model()
    
    test_texts = [
        "This is absolutely wonderful! Best experience ever!",
        "Terrible service. Very disappointed.",
        "It's okay, nothing special but works."
    ]
    
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"Text: '{text}'")
        print(f"➔ Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})\n")
    
    # 4. Entity Extraction
    print("\n4️⃣ ENTITY EXTRACTION")
    print("-"*40)
    extractor = EntityExtractor()
    extractor.load_model()
    
    entity_text = "Apple Inc. CEO Tim Cook announced the new iPhone in Cupertino, California."
    result = extractor.predict(entity_text)
    print(f"Text: '{entity_text}'")
    print("Entities found:")
    for entity in result['entities']:
        print(f"  • {entity['text']} ({entity['label']})")
    
    # 5. Full Pipeline Processing
    print("\n5️⃣ FULL PIPELINE PROCESSING")
    print("-"*40)
    pipeline = NLPPipeline()
    
    documents = [
        {"document_id": "doc1", "text": "Microsoft Azure provides excellent cloud services."},
        {"document_id": "doc2", "text": "The customer support was unhelpful and frustrating."},
        {"document_id": "doc3", "text": "Amazon Web Services announced new AI capabilities."}
    ]
    
    results = await pipeline.process_batch(documents)
    
    for result in results:
        print(f"\nDocument: {result.document_id}")
        print(f"Sentiment: {result.sentiment} ({result.sentiment_confidence:.2%})")
        print(f"Entities: {[e['text'] for e in result.entities]}")
        print(f"Processing time: {result.processing_time*1000:.1f}ms")
    
    # 6. Performance Metrics
    print("\n6️⃣ PERFORMANCE METRICS")
    print("-"*40)
    stats = pipeline.get_stats()
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Average processing time: {stats['average_processing_time']*1000:.1f}ms")
    print(f"Throughput: {stats['throughput']:.1f} docs/second")
    
    # Check if we met the performance target
    if stats['average_processing_time'] < 0.2:  # 200ms
        print("✅ Performance target met (< 200ms per document)")
    else:
        print("⚠️ Performance needs optimization")
    
    # 7. Monitoring
    print("\n7️⃣ MONITORING & HEALTH")
    print("-"*40)
    # Get metrics from Prometheus registry
    from prometheus_client import REGISTRY
    for collector in REGISTRY.collect():
        if collector.name == "documents_processed":
            for sample in collector.samples:
                print(f"Total documents processed: {int(sample.value)}")
        elif collector.name == "processing_errors":
            for sample in collector.samples:
                print(f"Processing errors: {int(sample.value)}")
    print("✅ Monitoring system active")
    
    # Shutdown
    await pipeline.shutdown()
    
    print("\n" + "="*80)
    print("✅ Feature showcase completed successfully!")
    print("="*80)


if __name__ == "__main__":
    # Run the showcase
    asyncio.run(showcase_features())