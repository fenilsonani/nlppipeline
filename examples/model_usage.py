"""Example usage of the NLP models module."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelManager, SentimentAnalyzer, EntityExtractor


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis."""
    print("\n=== Sentiment Analysis Demo ===")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Sample texts
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "The service was fantastic and the food was delicious!",
        "Worst experience ever. Would not recommend."
    ]
    
    # Load sentiment model
    sentiment_model = manager.load_model("sentiment")
    
    # Single prediction
    print("\nSingle text prediction:")
    result = sentiment_model.predict(texts[0])
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    print(f"Scores: {result['scores']}")
    
    # Batch prediction
    print("\nBatch predictions:")
    results = sentiment_model.predict_batch(texts)
    for result in results:
        print(f"- {result['text'][:50]}... -> {result['sentiment']} ({result['confidence']:.2f})")
    
    # Analyze distribution
    print("\nSentiment distribution:")
    distribution = sentiment_model.analyze_sentiment_distribution(texts)
    for key, value in distribution.items():
        print(f"{key}: {value}")


def demo_entity_extraction():
    """Demonstrate entity extraction."""
    print("\n=== Entity Extraction Demo ===")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Sample texts
    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Microsoft announced a new partnership with OpenAI on January 23, 2023.",
        "The meeting between Joe Biden and Emmanuel Macron will take place in Washington D.C.",
        "Google's CEO Sundar Pichai announced $100 million in AI research funding."
    ]
    
    # Load entity model
    entity_model = manager.load_model("entity")
    
    # Single text extraction
    print("\nSingle text entity extraction:")
    result = entity_model.predict(texts[0])
    print(f"Text: {result['text']}")
    print(f"Found {result['total_entities']} entities:")
    for entity in result['entities']:
        print(f"  - {entity['text']} ({entity['type']})")
    
    # Batch extraction
    print("\nBatch entity extraction:")
    results = entity_model.predict_batch(texts)
    for result in results:
        print(f"\nText: {result['text'][:60]}...")
        print(f"Entities: {', '.join([f'{e['text']} ({e['type']})' for e in result['entities']])}")
    
    # Extract specific entity types
    print("\nExtract specific entity types:")
    entity_types = entity_model.extract_entity_types(texts, ["Person", "Organization"])
    for entity_type, entities in entity_types.items():
        print(f"{entity_type}: {', '.join(entities)}")
    
    # Find relationships
    print("\nEntity relationships:")
    relationships = entity_model.find_entity_relationships(texts[0])
    for rel in relationships:
        print(f"{rel['entity1']['text']} --{rel['relation']}--> {rel['entity2']['text']}")


def demo_model_manager():
    """Demonstrate model manager features."""
    print("\n=== Model Manager Demo ===")
    
    # Initialize manager
    manager = ModelManager(max_models=3)
    
    # Load multiple models
    print("\nLoading models...")
    manager.load_model("sentiment")
    manager.load_model("entity")
    
    # List loaded models
    print("\nLoaded models:")
    for model_info in manager.list_loaded_models():
        print(f"- {model_info['key']}: {model_info['info']['name']} (loaded: {model_info['info']['is_loaded']})")
    
    # Direct prediction through manager
    print("\nDirect prediction through manager:")
    result = manager.predict("sentiment", "This is a great example!")
    print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2f})")
    
    # Batch prediction
    texts = ["Great product!", "Terrible service.", "It's okay."]
    results = manager.predict_batch("sentiment", texts)
    print("\nBatch predictions:")
    for result in results:
        print(f"- {result['text']} -> {result['sentiment']}")
    
    # Unload specific model
    print("\nUnloading sentiment model...")
    manager.unload_model("sentiment")
    
    print("\nRemaining models:")
    for model_info in manager.list_loaded_models():
        print(f"- {model_info['key']}")


def demo_combined_analysis():
    """Demonstrate combined sentiment and entity analysis."""
    print("\n=== Combined Analysis Demo ===")
    
    manager = ModelManager()
    
    # Sample news headlines
    headlines = [
        "Apple reports record profits, CEO Tim Cook celebrates successful quarter",
        "Tesla stock plummets after Elon Musk's controversial tweet",
        "Amazon announces major layoffs affecting 10,000 employees",
        "Microsoft's new AI product receives overwhelmingly positive reviews"
    ]
    
    print("\nAnalyzing news headlines...")
    
    for headline in headlines:
        print(f"\nHeadline: {headline}")
        
        # Sentiment analysis
        sentiment_result = manager.predict("sentiment", headline)
        print(f"Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.2f})")
        
        # Entity extraction
        entity_result = manager.predict("entity", headline)
        entities = [f"{e['text']} ({e['type']})" for e in entity_result['entities']]
        print(f"Entities: {', '.join(entities) if entities else 'None found'}")


if __name__ == "__main__":
    # Run all demos
    demo_sentiment_analysis()
    demo_entity_extraction()
    demo_model_manager()
    demo_combined_analysis()