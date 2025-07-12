"""
Example usage of the preprocessing module.
"""

from text_cleaner import TextCleaner, MultilingualTextCleaner
from tokenizer import BasicTokenizer, BERTTokenizer
from feature_extractor import TfidfFeatureExtractor, BERTFeatureExtractor
import numpy as np


def example_text_cleaning():
    """Demonstrate text cleaning functionality."""
    print("=== Text Cleaning Example ===")
    
    # Sample texts with noise
    texts = [
        "Check out https://example.com for more info! Email: user@example.com",
        "This is AMAZING!!! 😊 #NLP #TextProcessing",
        "Call me at 123-456-7890 or visit our website...",
        "I won't be there, but I'll send you the docs."
    ]
    
    # Basic text cleaner
    cleaner = TextCleaner(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_special_chars=True,
        remove_numbers=False
    )
    
    print("\nOriginal texts:")
    for text in texts:
        print(f"- {text}")
    
    print("\nCleaned texts:")
    cleaned_texts = cleaner.clean_batch(texts)
    for text in cleaned_texts:
        print(f"- {text}")
    
    # Multilingual cleaner with contraction expansion
    print("\n=== Multilingual Text Cleaning ===")
    ml_cleaner = MultilingualTextCleaner(language='en')
    
    contraction_text = "I won't go there. She'll be happy. They've arrived."
    print(f"\nOriginal: {contraction_text}")
    print(f"Expanded: {ml_cleaner.clean(contraction_text)}")
    
    return cleaned_texts


def example_tokenization(texts):
    """Demonstrate tokenization functionality."""
    print("\n\n=== Tokenization Example ===")
    
    # Basic tokenizer
    basic_tokenizer = BasicTokenizer(
        lowercase=True,
        remove_stopwords=True,
        min_token_length=3
    )
    
    print("\nBasic tokenization:")
    for text in texts[:2]:
        output = basic_tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {output.tokens}")
        print()
    
    # BERT tokenizer (if transformers is installed)
    try:
        print("\n=== BERT Tokenization ===")
        bert_tokenizer = BERTTokenizer(
            model_name='bert-base-uncased',
            max_length=128
        )
        
        sample_text = "Natural language processing is fascinating!"
        output = bert_tokenizer.tokenize(sample_text)
        
        print(f"Text: {sample_text}")
        print(f"Tokens: {output.tokens}")
        print(f"Token IDs: {output.token_ids}")
        print(f"Attention mask: {output.attention_mask}")
        
        # Decode back
        decoded = bert_tokenizer.decode(output.token_ids)
        print(f"Decoded: {decoded}")
        
    except Exception as e:
        print(f"BERT tokenization skipped: {e}")


def example_feature_extraction(texts):
    """Demonstrate feature extraction functionality."""
    print("\n\n=== Feature Extraction Example ===")
    
    # TF-IDF features
    print("\n1. TF-IDF Features:")
    tfidf_extractor = TfidfFeatureExtractor(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1
    )
    
    # Fit and extract
    tfidf_output = tfidf_extractor.fit_extract(texts)
    print(f"TF-IDF shape: {tfidf_output.features.shape}")
    print(f"Number of features: {len(tfidf_output.feature_names)}")
    print(f"Sample feature names: {tfidf_output.feature_names[:10]}")
    
    # Count vectorizer
    print("\n2. Count Features:")
    count_extractor = CountVectorFeatureExtractor(
        max_features=50,
        binary=True
    )
    
    count_extractor.fit(texts)
    count_output = count_extractor.extract(texts[0])
    print(f"Count features shape: {count_output.features.shape}")
    print(f"Non-zero features: {np.sum(count_output.features > 0)}")
    
    # Topic features (if we have enough documents)
    if len(texts) >= 10:
        print("\n3. Topic Features:")
        topic_extractor = TopicFeatureExtractor(
            n_topics=5,
            max_features=50
        )
        
        # Generate more sample texts for topic modeling
        extended_texts = texts * 3  # Simple repetition for demo
        topic_extractor.fit(extended_texts)
        topic_output = topic_extractor.extract(texts[0])
        
        print(f"Topic distribution shape: {topic_output.features.shape}")
        print(f"Topic probabilities: {topic_output.features[0]}")
        
        # Get top words per topic
        top_words = topic_extractor.get_top_words_per_topic(n_words=5)
        print("\nTop words per topic:")
        for topic_id, words in top_words.items():
            print(f"Topic {topic_id}: {', '.join(words)}")
    
    # BERT features (if available)
    try:
        print("\n4. BERT Features:")
        bert_extractor = BERTFeatureExtractor(
            model_name='bert-base-uncased',
            pooling_strategy='mean'
        )
        
        bert_output = bert_extractor.extract(texts[:2])
        print(f"BERT embeddings shape: {bert_output.features.shape}")
        print(f"Embedding dimension: {bert_output.metadata['embedding_dim']}")
        
    except Exception as e:
        print(f"BERT feature extraction skipped: {e}")


def example_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n\n=== Batch Processing Example ===")
    
    # Generate sample data
    large_texts = [
        f"This is sample text number {i} for batch processing demonstration."
        for i in range(100)
    ]
    
    # Batch cleaning
    cleaner = TextCleaner()
    cleaned = cleaner.clean_batch(large_texts, batch_size=25)
    print(f"Cleaned {len(cleaned)} texts in batches")
    
    # Batch tokenization with BERT
    try:
        bert_tokenizer = BERTTokenizer()
        outputs = bert_tokenizer.tokenize_batch(large_texts[:10], batch_size=5)
        print(f"Tokenized {len(outputs)} texts using BERT")
    except:
        print("BERT batch tokenization skipped")


def main():
    """Run all examples."""
    # Text cleaning example
    cleaned_texts = example_text_cleaning()
    
    # Add more sample texts for other examples
    sample_texts = cleaned_texts + [
        "Machine learning algorithms can process natural language efficiently.",
        "Deep learning models like BERT have revolutionized NLP tasks.",
        "Text preprocessing is crucial for model performance.",
        "Feature extraction transforms text into numerical representations.",
        "Tokenization splits text into meaningful units.",
        "Stop words are commonly removed during preprocessing."
    ]
    
    # Tokenization example
    example_tokenization(sample_texts)
    
    # Feature extraction example
    example_feature_extraction(sample_texts)
    
    # Batch processing example
    example_batch_processing()
    
    print("\n\n=== Preprocessing Module Demo Complete ===")


if __name__ == "__main__":
    main()