# Preprocessing Module

This module provides comprehensive text preprocessing utilities for NLP pipelines, including text cleaning, tokenization, and feature extraction.

## Components

### 1. Text Cleaner (`text_cleaner.py`)

Provides text cleaning and normalization functionality:

- **TextCleaner**: Basic text cleaning with configurable options
  - Remove URLs, emails, special characters
  - Lowercase conversion
  - Number and punctuation removal
  - Whitespace normalization
  - Custom stopword removal
  - Batch processing support

- **MultilingualTextCleaner**: Extended cleaner with language-specific features
  - Contraction expansion (e.g., "don't" → "do not")
  - Language-specific pattern handling
  - Inherits all TextCleaner functionality

### 2. Tokenizer (`tokenizer.py`)

Text tokenization utilities:

- **BasicTokenizer**: NLTK-based word and sentence tokenization
  - Configurable stopword removal
  - Token length filtering
  - Sentence splitting

- **BERTTokenizer**: Transformer-based tokenization
  - Support for any Hugging Face model
  - Automatic special token handling
  - Batch processing with configurable batch size
  - Token ID encoding/decoding

- **SubwordTokenizer**: Placeholder for BPE/WordPiece tokenization

### 3. Feature Extractor (`feature_extractor.py`)

Feature extraction for ML models:

- **TfidfFeatureExtractor**: TF-IDF based features
  - Configurable n-gram ranges
  - Document frequency filtering
  - Sparse to dense conversion

- **BERTFeatureExtractor**: Dense embeddings from transformer models
  - Multiple pooling strategies (CLS, mean, max)
  - GPU support with automatic device selection
  - Batch processing for efficiency
  - Optional caching for repeated texts

- **CountVectorFeatureExtractor**: Bag-of-words features
  - Binary or count-based features
  - N-gram support

- **TopicFeatureExtractor**: LDA-based topic modeling
  - Configurable number of topics
  - Topic word extraction

- **EmbeddingFeatureExtractor**: Generic word embedding aggregation

## Usage Examples

### Basic Text Cleaning

```python
from preprocessing import TextCleaner

cleaner = TextCleaner(
    lowercase=True,
    remove_urls=True,
    remove_emails=True,
    remove_special_chars=True
)

text = "Check out https://example.com! Email: user@example.com"
cleaned = cleaner.clean(text)
# Output: "check out email"

# Batch processing
texts = ["Text 1 with URL https://example.com", "Text 2 with numbers 123"]
cleaned_texts = cleaner.clean_batch(texts)
```

### Multilingual Cleaning

```python
from preprocessing import MultilingualTextCleaner

ml_cleaner = MultilingualTextCleaner(language='en')
text = "I won't be there, but I'll send the docs."
cleaned = ml_cleaner.clean(text)
# Output: "i will not be there but i will send the docs"
```

### Tokenization

```python
from preprocessing import BasicTokenizer, BERTTokenizer

# Basic tokenization
basic_tokenizer = BasicTokenizer(remove_stopwords=True)
output = basic_tokenizer.tokenize("This is a sample text for tokenization.")
print(output.tokens)  # ['sample', 'text', 'tokenization']

# BERT tokenization
bert_tokenizer = BERTTokenizer(model_name='bert-base-uncased')
output = bert_tokenizer.tokenize("Natural language processing")
print(output.tokens)  # ['[CLS]', 'natural', 'language', 'processing', '[SEP]']
print(output.token_ids)  # [101, 3019, 2653, 6364, 102]
```

### Feature Extraction

```python
from preprocessing import TfidfFeatureExtractor, BERTFeatureExtractor

# TF-IDF features
tfidf = TfidfFeatureExtractor(max_features=1000, ngram_range=(1, 2))
tfidf.fit(train_texts)
features = tfidf.extract(test_texts)
print(features.features.shape)  # (n_texts, 1000)

# BERT embeddings
bert_extractor = BERTFeatureExtractor(
    model_name='bert-base-uncased',
    pooling_strategy='mean'
)
embeddings = bert_extractor.extract(texts)
print(embeddings.features.shape)  # (n_texts, 768)
```

## Batch Processing

All components support batch processing for handling large datasets efficiently:

```python
# Process 10,000 texts in batches
large_corpus = ["text " + str(i) for i in range(10000)]

# Batch cleaning
cleaner = TextCleaner()
cleaned = cleaner.clean_batch(large_corpus, batch_size=1000)

# Batch tokenization
tokenizer = BERTTokenizer()
token_outputs = tokenizer.tokenize_batch(cleaned, batch_size=32)

# Batch feature extraction
extractor = BERTFeatureExtractor(batch_size=32)
features = extractor.extract(cleaned)
```

## Performance Considerations

1. **Caching**: Use `@lru_cache` for repeated operations
2. **Batch Processing**: Process multiple texts together for better performance
3. **GPU Usage**: BERT-based components automatically use GPU if available
4. **Memory Management**: Use generators for very large datasets
5. **Preprocessing Pipeline**: Chain operations efficiently

## Dependencies

- `nltk`: Basic NLP operations
- `transformers`: Transformer models and tokenizers
- `torch`: Deep learning backend
- `scikit-learn`: Traditional ML feature extraction
- `numpy`: Numerical operations

## Language Support

Currently focused on English with extensible support for other languages through:
- Language-specific text cleaners
- Multilingual transformer models
- Configurable stopword lists