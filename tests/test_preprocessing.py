"""Comprehensive tests for preprocessing modules."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.preprocessing.text_cleaner import TextCleaner, MultilingualTextCleaner
from src.preprocessing.tokenizer import (
    BasicTokenizer, BERTTokenizer, SubwordTokenizer,
    TokenizationOutput
)


class TestTextCleaner:
    """Test suite for TextCleaner class."""
    
    def test_init_default_options(self):
        """Test TextCleaner initialization with default options."""
        cleaner = TextCleaner()
        assert cleaner.lowercase is True
        assert cleaner.remove_urls is True
        assert cleaner.remove_emails is True
        assert cleaner.remove_special_chars is True
        assert cleaner.remove_numbers is False
        assert cleaner.remove_punctuation is False
        assert cleaner.remove_extra_whitespace is True
        assert cleaner.normalize_unicode is True
        assert cleaner.custom_stopwords == set()
    
    def test_init_custom_options(self):
        """Test TextCleaner initialization with custom options."""
        custom_stopwords = {"the", "and", "or"}
        cleaner = TextCleaner(
            lowercase=False,
            remove_urls=False,
            remove_numbers=True,
            custom_stopwords=custom_stopwords
        )
        assert cleaner.lowercase is False
        assert cleaner.remove_urls is False
        assert cleaner.remove_numbers is True
        assert cleaner.custom_stopwords == custom_stopwords
    
    def test_clean_basic_text(self, sample_text):
        """Test basic text cleaning functionality."""
        cleaner = TextCleaner()
        result = cleaner.clean(sample_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == result.lower()  # Should be lowercase
    
    def test_clean_empty_text(self):
        """Test cleaning empty or whitespace-only text."""
        cleaner = TextCleaner()
        
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""
        assert cleaner.clean("\n\t ") == ""
    
    def test_clean_non_string_input(self):
        """Test cleaning non-string input."""
        cleaner = TextCleaner()
        
        assert cleaner.clean(None) == ""
        assert cleaner.clean(123) == ""
        assert cleaner.clean([]) == ""
    
    def test_url_removal(self):
        """Test URL removal functionality."""
        cleaner = TextCleaner(remove_urls=True)
        text = "Check out https://example.com and http://test.org for more info"
        result = cleaner.clean(text)
        
        assert "https://example.com" not in result
        assert "http://test.org" not in result
        assert "check out" in result
        assert "for more info" in result
    
    def test_url_preservation(self):
        """Test URL preservation when disabled."""
        cleaner = TextCleaner(remove_urls=False)
        text = "Visit https://example.com"
        result = cleaner.clean(text)
        
        assert "https://example.com" in result
    
    def test_email_removal(self):
        """Test email removal functionality."""
        cleaner = TextCleaner(remove_emails=True)
        text = "Contact me at user@domain.com or admin@site.org"
        result = cleaner.clean(text)
        
        assert "user@domain.com" not in result
        assert "admin@site.org" not in result
        assert "contact me at" in result
    
    def test_email_preservation(self):
        """Test email preservation when disabled."""
        cleaner = TextCleaner(remove_emails=False)
        text = "Email user@domain.com"
        result = cleaner.clean(text)
        
        assert "user@domain.com" in result
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        text = "UPPERCASE and MiXeD cAsE"
        result = cleaner.clean(text)
        
        assert result == "uppercase and mixed case"
    
    def test_lowercase_preservation(self):
        """Test case preservation when disabled."""
        cleaner = TextCleaner(lowercase=False)
        text = "UPPERCASE and MiXeD cAsE"
        result = cleaner.clean(text)
        
        assert "UPPERCASE" in result
        assert "MiXeD" in result
    
    def test_number_removal(self):
        """Test number removal functionality."""
        cleaner = TextCleaner(remove_numbers=True)
        text = "I have 5 apples and 10 oranges, total 123 fruits"
        result = cleaner.clean(text)
        
        assert "5" not in result
        assert "10" not in result
        assert "123" not in result
        assert "apples" in result
        assert "oranges" in result
    
    def test_punctuation_removal(self):
        """Test punctuation removal functionality."""
        cleaner = TextCleaner(remove_punctuation=True)
        text = "Hello, world! How are you? I'm fine."
        result = cleaner.clean(text)
        
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert "'" not in result
        assert "hello world" in result
    
    def test_special_chars_removal(self):
        """Test special characters removal."""
        cleaner = TextCleaner(remove_special_chars=True, remove_punctuation=False)
        text = "Price: $19.99 @mention #hashtag"
        result = cleaner.clean(text)
        
        assert "$" not in result
        assert "@" not in result
        assert "#" not in result
        assert "price" in result
        assert "mention" in result
        assert "hashtag" in result
    
    def test_whitespace_normalization(self):
        """Test extra whitespace removal."""
        cleaner = TextCleaner(remove_extra_whitespace=True)
        text = "Too    much     whitespace   here"
        result = cleaner.clean(text)
        
        assert "too much whitespace here" == result
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        cleaner = TextCleaner(normalize_unicode=True)
        text = "café naïve résumé"
        result = cleaner.clean(text)
        
        # Should normalize accented characters
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_custom_stopwords_removal(self):
        """Test custom stopwords removal."""
        stopwords = {"the", "and", "or"}
        cleaner = TextCleaner(custom_stopwords=stopwords)
        text = "The cat and the dog or the bird"
        result = cleaner.clean(text)
        
        words = result.split()
        assert "the" not in words
        assert "and" not in words
        assert "or" not in words
        assert "cat" in words
        assert "dog" in words
        assert "bird" in words
    
    def test_clean_batch(self, sample_texts):
        """Test batch cleaning functionality."""
        cleaner = TextCleaner()
        results = cleaner.clean_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, str) for r in results)
        
        # Verify each text was cleaned
        for original, cleaned in zip(sample_texts, results):
            if original.strip():  # Skip empty strings
                assert len(cleaned) <= len(original)
    
    def test_clean_batch_with_batch_size(self, large_text_batch):
        """Test batch cleaning with custom batch size."""
        cleaner = TextCleaner()
        results = cleaner.clean_batch(large_text_batch, batch_size=50)
        
        assert len(results) == len(large_text_batch)
        assert all(isinstance(r, str) for r in results)
    
    @pytest.mark.performance
    def test_cleaning_performance(self, large_text_batch, performance_timer):
        """Test text cleaning performance."""
        cleaner = TextCleaner()
        
        performance_timer.start()
        results = cleaner.clean_batch(large_text_batch)
        elapsed = performance_timer.stop()
        
        assert elapsed < 5.0  # Should complete within 5 seconds
        assert len(results) == len(large_text_batch)
        
        # Calculate throughput
        throughput = len(large_text_batch) / elapsed
        assert throughput > 100  # Should process >100 texts per second
    
    def test_remove_stopwords_static_method(self):
        """Test static stopwords removal method."""
        text = "this is a test document"
        stopwords = frozenset(["this", "is", "a"])
        
        result = TextCleaner.remove_stopwords(text, stopwords)
        assert result == "test document"
    
    def test_remove_stopwords_caching(self):
        """Test LRU cache functionality for stopwords removal."""
        text = "this is a test"
        stopwords = frozenset(["this", "is"])
        
        # Call multiple times to test caching
        result1 = TextCleaner.remove_stopwords(text, stopwords)
        result2 = TextCleaner.remove_stopwords(text, stopwords)
        
        assert result1 == result2 == "a test"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        cleaner = TextCleaner()
        
        # Very long text
        long_text = "a" * 10000
        result = cleaner.clean(long_text)
        assert isinstance(result, str)
        
        # Text with only special characters
        special_text = "@#$%^&*()"
        result = cleaner.clean(special_text)
        assert len(result) == 0 or result.isspace()
        
        # Mixed Unicode text
        unicode_text = "Hello 世界 🌍"
        result = cleaner.clean(unicode_text)
        assert isinstance(result, str)


class TestMultilingualTextCleaner:
    """Test suite for MultilingualTextCleaner class."""
    
    def test_init_english(self):
        """Test initialization for English language."""
        cleaner = MultilingualTextCleaner(language='en')
        assert cleaner.language == 'en'
        assert len(cleaner.contraction_patterns) > 0
    
    def test_init_other_language(self):
        """Test initialization for non-English language."""
        cleaner = MultilingualTextCleaner(language='es')
        assert cleaner.language == 'es'
        assert len(cleaner.contraction_patterns) == 0
    
    def test_expand_contractions(self):
        """Test contraction expansion for English."""
        cleaner = MultilingualTextCleaner(language='en')
        text = "I can't believe you won't come. I'm disappointed."
        result = cleaner.expand_contractions(text)
        
        assert "cannot" in result
        assert "will not" in result
        assert "I am" in result
        assert "can't" not in result
        assert "won't" not in result
        assert "I'm" not in result
    
    def test_clean_with_contractions(self):
        """Test cleaning with contraction expansion."""
        cleaner = MultilingualTextCleaner(language='en')
        text = "I won't go there. It's not possible."
        result = cleaner.clean(text)
        
        assert "will not" in result
        assert "it is" in result
    
    def test_clean_non_english(self):
        """Test cleaning for non-English language."""
        cleaner = MultilingualTextCleaner(language='es')
        text = "Hola mundo. ¿Cómo estás?"
        result = cleaner.clean(text)
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestBasicTokenizer:
    """Test suite for BasicTokenizer class."""
    
    def test_init_default_options(self):
        """Test BasicTokenizer initialization with defaults."""
        tokenizer = BasicTokenizer()
        assert tokenizer.lowercase is True
        assert tokenizer.remove_stopwords is False
        assert tokenizer.min_token_length == 1
        assert tokenizer.max_token_length is None
    
    def test_init_custom_options(self):
        """Test BasicTokenizer initialization with custom options."""
        tokenizer = BasicTokenizer(
            lowercase=False,
            remove_stopwords=True,
            language='spanish',
            min_token_length=2,
            max_token_length=10
        )
        assert tokenizer.lowercase is False
        assert tokenizer.remove_stopwords is True
        assert tokenizer.min_token_length == 2
        assert tokenizer.max_token_length == 10
    
    def test_tokenize_basic(self, sample_text):
        """Test basic tokenization."""
        tokenizer = BasicTokenizer()
        result = tokenizer.tokenize(sample_text)
        
        assert isinstance(result, TokenizationOutput)
        assert isinstance(result.tokens, list)
        assert len(result.tokens) > 0
        assert all(isinstance(token, str) for token in result.tokens)
    
    def test_tokenize_lowercase(self):
        """Test lowercase tokenization."""
        tokenizer = BasicTokenizer(lowercase=True)
        text = "Hello WORLD"
        result = tokenizer.tokenize(text)
        
        assert "hello" in result.tokens
        assert "world" in result.tokens
        assert "Hello" not in result.tokens
        assert "WORLD" not in result.tokens
    
    def test_tokenize_preserve_case(self):
        """Test case preservation."""
        tokenizer = BasicTokenizer(lowercase=False)
        text = "Hello WORLD"
        result = tokenizer.tokenize(text)
        
        assert "Hello" in result.tokens
        assert "WORLD" in result.tokens
    
    @patch('nltk.corpus.stopwords.words')
    def test_tokenize_remove_stopwords(self, mock_stopwords):
        """Test stopword removal."""
        mock_stopwords.return_value = ['the', 'is', 'a']
        
        tokenizer = BasicTokenizer(remove_stopwords=True)
        text = "The cat is a pet"
        result = tokenizer.tokenize(text)
        
        tokens_lower = [t.lower() for t in result.tokens]
        assert 'the' not in tokens_lower
        assert 'is' not in tokens_lower
        assert 'a' not in tokens_lower
        assert 'cat' in tokens_lower
        assert 'pet' in tokens_lower
    
    def test_tokenize_min_length_filter(self):
        """Test minimum token length filtering."""
        tokenizer = BasicTokenizer(min_token_length=3)
        text = "I am a big cat"
        result = tokenizer.tokenize(text)
        
        # Only tokens with 3+ characters should remain
        valid_tokens = [t for t in result.tokens if len(t) >= 3]
        assert result.tokens == valid_tokens
        assert "big" in result.tokens
        assert "cat" in result.tokens
    
    def test_tokenize_max_length_filter(self):
        """Test maximum token length filtering."""
        tokenizer = BasicTokenizer(max_token_length=5)
        text = "short verylongword medium"
        result = tokenizer.tokenize(text)
        
        # Only tokens with 5 or fewer characters should remain
        assert all(len(token) <= 5 for token in result.tokens)
        assert "short" in result.tokens
        assert "verylongword" not in result.tokens
    
    def test_tokenize_batch(self, sample_texts):
        """Test batch tokenization."""
        tokenizer = BasicTokenizer()
        results = tokenizer.tokenize_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, TokenizationOutput) for r in results)
        
        # Check that non-empty texts produce tokens
        for i, result in enumerate(results):
            if sample_texts[i].strip():
                assert len(result.tokens) > 0
    
    def test_tokenize_sentences(self, sample_text):
        """Test sentence tokenization."""
        tokenizer = BasicTokenizer()
        sentences = tokenizer.tokenize_sentences(sample_text)
        
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        assert all(isinstance(s, str) for s in sentences)
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        tokenizer = BasicTokenizer()
        result = tokenizer.tokenize("")
        
        assert isinstance(result, TokenizationOutput)
        assert len(result.tokens) == 0
    
    @pytest.mark.performance
    def test_tokenization_performance(self, large_text_batch, performance_timer):
        """Test tokenization performance."""
        tokenizer = BasicTokenizer()
        
        performance_timer.start()
        results = tokenizer.tokenize_batch(large_text_batch)
        elapsed = performance_timer.stop()
        
        assert elapsed < 10.0  # Should complete within 10 seconds
        assert len(results) == len(large_text_batch)
        
        # Calculate throughput
        throughput = len(large_text_batch) / elapsed
        assert throughput > 50  # Should process >50 texts per second


class TestBERTTokenizer:
    """Test suite for BERTTokenizer class."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_init(self, mock_tokenizer):
        """Test BERTTokenizer initialization."""
        mock_tokenizer.return_value = Mock()
        
        tokenizer = BERTTokenizer(
            model_name='bert-base-uncased',
            max_length=256,
            padding=True,
            truncation=True
        )
        
        assert tokenizer.model_name == 'bert-base-uncased'
        assert tokenizer.max_length == 256
        assert tokenizer.padding is True
        assert tokenizer.truncation is True
        mock_tokenizer.assert_called_once_with('bert-base-uncased')
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize(self, mock_from_pretrained):
        """Test BERT tokenization."""
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [101, 7592, 2088, 102],  # [CLS] hello world [SEP]
            'attention_mask': [1, 1, 1, 1]
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'hello', 'world', '[SEP]']
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        result = tokenizer.tokenize("hello world")
        
        assert isinstance(result, TokenizationOutput)
        assert result.tokens == ['[CLS]', 'hello', 'world', '[SEP]']
        assert result.token_ids == [101, 7592, 2088, 102]
        assert result.attention_mask == [1, 1, 1, 1]
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_with_tensors(self, mock_from_pretrained):
        """Test BERT tokenization with tensor output."""
        # Mock the tokenizer with tensor output
        mock_tensor = Mock()
        mock_tensor.tolist.return_value = [101, 7592, 2088, 102]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [mock_tensor],
            'attention_mask': [mock_tensor]
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'hello', 'world', '[SEP]']
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer(return_tensors='pt')
        result = tokenizer.tokenize("hello world")
        
        assert isinstance(result, TokenizationOutput)
        assert isinstance(result.tokens, list)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_batch(self, mock_from_pretrained):
        """Test BERT batch tokenization."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[101, 7592, 102], [101, 2088, 102]],
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        mock_tokenizer.convert_ids_to_tokens.side_effect = [
            ['[CLS]', 'hello', '[SEP]'],
            ['[CLS]', 'world', '[SEP]']
        ]
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        texts = ["hello", "world"]
        results = tokenizer.tokenize_batch(texts)
        
        assert len(results) == 2
        assert all(isinstance(r, TokenizationOutput) for r in results)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_decode(self, mock_from_pretrained):
        """Test token decoding."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "hello world"
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        result = tokenizer.decode([101, 7592, 2088, 102])
        
        assert result == "hello world"
        mock_tokenizer.decode.assert_called_once_with([101, 7592, 2088, 102], skip_special_tokens=True)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_vocab_size_property(self, mock_from_pretrained):
        """Test vocabulary size property."""
        mock_tokenizer = Mock()
        mock_tokenizer.__len__.return_value = 30522
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        assert tokenizer.vocab_size == 30522
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_special_tokens_property(self, mock_from_pretrained):
        """Test special tokens property."""
        mock_tokenizer = Mock()
        mock_tokenizer.cls_token = '[CLS]'
        mock_tokenizer.sep_token = '[SEP]'
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer.unk_token = '[UNK]'
        mock_tokenizer.mask_token = '[MASK]'
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        special_tokens = tokenizer.special_tokens
        
        assert special_tokens['cls_token'] == '[CLS]'
        assert special_tokens['sep_token'] == '[SEP]'
        assert special_tokens['pad_token'] == '[PAD]'
        assert special_tokens['unk_token'] == '[UNK]'
        assert special_tokens['mask_token'] == '[MASK]'
    
    @pytest.mark.performance
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_batch_tokenization_performance(self, mock_from_pretrained, large_text_batch, performance_timer):
        """Test BERT batch tokenization performance."""
        # Mock fast tokenization
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[101, 102]] * len(large_text_batch),
            'attention_mask': [[1, 1]] * len(large_text_batch)
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', '[SEP]']
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        
        performance_timer.start()
        results = tokenizer.tokenize_batch(large_text_batch, batch_size=64)
        elapsed = performance_timer.stop()
        
        assert elapsed < 15.0  # Should complete within 15 seconds
        assert len(results) == len(large_text_batch)


class TestSubwordTokenizer:
    """Test suite for SubwordTokenizer class."""
    
    def test_init(self):
        """Test SubwordTokenizer initialization."""
        tokenizer = SubwordTokenizer(vocab_size=20000, min_frequency=5)
        assert tokenizer.vocab_size == 20000
        assert tokenizer.min_frequency == 5
        assert tokenizer.tokenizer is None
    
    def test_train_not_implemented(self):
        """Test that training raises NotImplementedError."""
        tokenizer = SubwordTokenizer()
        with pytest.raises(NotImplementedError):
            tokenizer.train(["sample text"])
    
    def test_tokenize_without_training(self):
        """Test tokenization without training raises error."""
        tokenizer = SubwordTokenizer()
        with pytest.raises(ValueError, match="Tokenizer not trained"):
            tokenizer.tokenize("test text")


class TestTokenizationOutput:
    """Test suite for TokenizationOutput dataclass."""
    
    def test_creation_minimal(self):
        """Test TokenizationOutput creation with minimal fields."""
        output = TokenizationOutput(tokens=['hello', 'world'])
        assert output.tokens == ['hello', 'world']
        assert output.token_ids is None
        assert output.attention_mask is None
        assert output.token_type_ids is None
        assert output.special_tokens_mask is None
    
    def test_creation_full(self):
        """Test TokenizationOutput creation with all fields."""
        output = TokenizationOutput(
            tokens=['[CLS]', 'hello', '[SEP]'],
            token_ids=[101, 7592, 102],
            attention_mask=[1, 1, 1],
            token_type_ids=[0, 0, 0],
            special_tokens_mask=[1, 0, 1]
        )
        assert output.tokens == ['[CLS]', 'hello', '[SEP]']
        assert output.token_ids == [101, 7592, 102]
        assert output.attention_mask == [1, 1, 1]
        assert output.token_type_ids == [0, 0, 0]
        assert output.special_tokens_mask == [1, 0, 1]


# Integration tests
class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""
    
    def test_cleaner_tokenizer_pipeline(self, noisy_texts):
        """Test text cleaner and tokenizer integration."""
        cleaner = TextCleaner()
        tokenizer = BasicTokenizer()
        
        for text in noisy_texts:
            if isinstance(text, str) and text.strip():
                # Clean then tokenize
                cleaned = cleaner.clean(text)
                result = tokenizer.tokenize(cleaned)
                
                assert isinstance(result, TokenizationOutput)
                assert isinstance(result.tokens, list)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_multilingual_cleaner_bert_tokenizer(self, mock_from_pretrained, multilingual_texts):
        """Test multilingual cleaner with BERT tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [101, 102],
            'attention_mask': [1, 1]
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', '[SEP]']
        mock_from_pretrained.return_value = mock_tokenizer
        
        cleaner = MultilingualTextCleaner(language='en')
        tokenizer = BERTTokenizer()
        
        for text in multilingual_texts:
            cleaned = cleaner.clean(text)
            result = tokenizer.tokenize(cleaned)
            
            assert isinstance(result, TokenizationOutput)
            assert len(result.tokens) >= 2  # At least [CLS] and [SEP]
    
    @pytest.mark.performance
    def test_full_preprocessing_pipeline_performance(self, large_text_batch, performance_timer):
        """Test performance of full preprocessing pipeline."""
        cleaner = TextCleaner()
        tokenizer = BasicTokenizer()
        
        performance_timer.start()
        
        # Process full pipeline
        cleaned_batch = cleaner.clean_batch(large_text_batch)
        tokenized_batch = tokenizer.tokenize_batch(cleaned_batch)
        
        elapsed = performance_timer.stop()
        
        assert elapsed < 20.0  # Should complete within 20 seconds
        assert len(tokenized_batch) == len(large_text_batch)
        
        # Calculate throughput
        throughput = len(large_text_batch) / elapsed
        assert throughput > 25  # Should process >25 texts per second through full pipeline


# Error handling tests
class TestPreprocessingErrorHandling:
    """Test error handling in preprocessing components."""
    
    def test_text_cleaner_error_handling(self, error_prone_texts):
        """Test text cleaner error handling."""
        cleaner = TextCleaner()
        
        for text in error_prone_texts:
            try:
                result = cleaner.clean(text)
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"TextCleaner should handle input {text}, but raised {e}")
    
    def test_basic_tokenizer_error_handling(self, error_prone_texts):
        """Test basic tokenizer error handling."""
        tokenizer = BasicTokenizer()
        
        for text in error_prone_texts:
            if isinstance(text, str):
                try:
                    result = tokenizer.tokenize(text)
                    assert isinstance(result, TokenizationOutput)
                except Exception as e:
                    pytest.fail(f"BasicTokenizer should handle input {text}, but raised {e}")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_bert_tokenizer_error_handling(self, mock_from_pretrained):
        """Test BERT tokenizer error handling."""
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = BERTTokenizer()
        
        with pytest.raises(Exception):
            tokenizer.tokenize("test text")


# Edge case tests
class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing."""
    
    def test_extremely_long_text(self):
        """Test processing extremely long text."""
        cleaner = TextCleaner()
        tokenizer = BasicTokenizer()
        
        # Create 100KB text
        long_text = "This is a test sentence. " * 4000
        
        cleaned = cleaner.clean(long_text)
        result = tokenizer.tokenize(cleaned)
        
        assert isinstance(cleaned, str)
        assert isinstance(result, TokenizationOutput)
        assert len(result.tokens) > 0
    
    def test_unicode_edge_cases(self):
        """Test Unicode edge cases."""
        cleaner = TextCleaner()
        tokenizer = BasicTokenizer()
        
        unicode_texts = [
            "🚀🌟💫🎉🔥",  # Emojis only
            "数据科学很有趣",  # Chinese characters
            "Ελληνικά",  # Greek
            "العربية",  # Arabic
            "עברית",  # Hebrew
            "🇺🇸🇬🇧🇫🇷",  # Flag emojis
        ]
        
        for text in unicode_texts:
            cleaned = cleaner.clean(text)
            result = tokenizer.tokenize(cleaned)
            
            assert isinstance(cleaned, str)
            assert isinstance(result, TokenizationOutput)
    
    def test_mixed_content_edge_cases(self):
        """Test mixed content edge cases."""
        cleaner = TextCleaner()
        
        mixed_texts = [
            "Text with\nline\nbreaks",
            "Text\twith\ttabs",
            "Text\r\nwith\r\nCRLF",
            "Text with\x00null\x01chars",
            "Multiple   spaces    and   tabs\t\t",
        ]
        
        for text in mixed_texts:
            result = cleaner.clean(text)
            assert isinstance(result, str)
            # Should normalize whitespace
            assert "\n" not in result or not cleaner.remove_extra_whitespace
            assert "\t" not in result or not cleaner.remove_extra_whitespace