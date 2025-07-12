#!/usr/bin/env python3
"""
NLP Pipeline Demo Script

This script demonstrates the key features of the Enterprise-Grade NLP Pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import NLPPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Run NLP Pipeline demo."""
    print("=" * 80)
    print("NLP Pipeline Demo - Enterprise-Grade Sentiment Analysis & Entity Extraction")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n1. Initializing NLP Pipeline...")
    pipeline = NLPPipeline()
    print("✓ Pipeline initialized successfully")
    
    # Demo documents
    demo_documents = [
        {
            "document_id": "demo_001",
            "text": "Apple Inc. announced exceptional quarterly results. The iPhone sales exceeded expectations!"
        },
        {
            "document_id": "demo_002", 
            "text": "The customer service at Amazon was disappointing. Long wait times and unhelpful responses."
        },
        {
            "document_id": "demo_003",
            "text": "Microsoft Azure provides reliable cloud infrastructure for enterprise applications."
        },
        {
            "document_id": "demo_004",
            "text": "Google's new AI model shows promising results in natural language understanding."
        }
    ]
    
    # Process documents
    print("\n2. Processing documents...")
    results = await pipeline.process_batch(demo_documents)
    
    # Display results
    print("\n3. Analysis Results:")
    print("-" * 80)
    
    for result in results:
        print(f"\nDocument ID: {result.document_id}")
        print(f"Text: {result.text[:80]}...")
        print(f"Sentiment: {result.sentiment} (Confidence: {result.sentiment_confidence:.2%})")
        
        if result.entities:
            print("Entities found:")
            for entity in result.entities:
                print(f"  - {entity['text']} ({entity['label']})")
        else:
            print("Entities: None found")
        
        print(f"Processing time: {result.processing_time*1000:.1f}ms")
    
    # Show statistics
    print("\n4. Performance Statistics:")
    print("-" * 80)
    stats = pipeline.get_stats()
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Average processing time: {stats['average_processing_time']*1000:.1f}ms")
    print(f"Throughput: {stats['throughput']:.1f} documents/second")
    
    # Show aggregated results
    agg = stats['aggregated_results']
    print(f"\nSentiment Distribution:")
    if agg['sentiment'] and agg['sentiment']['distribution']:
        for sentiment, count in agg['sentiment']['distribution'].items():
            percentage = (count / agg['total_documents']) * 100 if agg['total_documents'] > 0 else 0
            print(f"  - {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"\nTotal entities extracted: {agg['entities']['total']}")
    print(f"Unique entities: {agg['entities']['unique']}")
    
    # Export results
    print("\n5. Exporting results...")
    await pipeline._export_results()
    print(f"✓ Results exported to: {pipeline.config.storage.output_path}")
    
    # Shutdown
    await pipeline.shutdown()
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise