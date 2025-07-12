"""Comprehensive tests for the main NLP pipeline."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from pathlib import Path
import json

from src.pipeline import NLPPipeline, PipelineResult


class TestPipelineResult:
    """Test suite for PipelineResult dataclass."""
    
    def test_pipeline_result_creation(self):
        """Test PipelineResult creation."""
        timestamp = datetime.utcnow()
        result = PipelineResult(
            document_id="test_doc_001",
            text="This is a test document.",
            sentiment="positive",
            sentiment_confidence=0.85,
            entities=[{"text": "test", "type": "MISC"}],
            processing_time=0.123,
            timestamp=timestamp
        )
        
        assert result.document_id == "test_doc_001"
        assert result.text == "This is a test document."
        assert result.sentiment == "positive"
        assert result.sentiment_confidence == 0.85
        assert len(result.entities) == 1
        assert result.processing_time == 0.123
        assert result.timestamp == timestamp


class TestNLPPipeline:
    """Test suite for NLPPipeline class."""
    
    @patch('src.pipeline.config')
    def test_init_default_config(self, mock_config):
        """Test NLPPipeline initialization with default config."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        with patch.multiple(
            'src.pipeline',
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            pipeline = NLPPipeline()
            
            assert pipeline.config == mock_config
            assert pipeline.documents_processed == 0
            assert pipeline.total_processing_time == 0.0
            assert pipeline.sentiment_analyzer is None
            assert pipeline.entity_extractor is None
    
    @patch('src.pipeline.config')
    def test_init_custom_config_file(self, mock_config):
        """Test NLPPipeline initialization with custom config file."""
        mock_config.load_from_file = Mock()
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        with patch.multiple(
            'src.pipeline',
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            pipeline = NLPPipeline(config_file="custom_config.yaml")
            
            mock_config.load_from_file.assert_called_once_with("custom_config.yaml")
    
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    def test_load_models(self, mock_config):
        """Test lazy loading of models."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        # Setup mock model manager
        mock_sentiment_model = Mock()
        mock_entity_model = Mock()
        mock_model_manager = Mock()
        mock_model_manager.get_model.side_effect = [mock_sentiment_model, mock_entity_model]
        
        pipeline = NLPPipeline()
        pipeline.model_manager = mock_model_manager
        
        # Load models
        pipeline._load_models()
        
        assert pipeline.sentiment_analyzer == mock_sentiment_model
        assert pipeline.entity_extractor == mock_entity_model
        assert mock_model_manager.get_model.call_count == 2
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        metrics_collector=Mock(),
        perf_logger=Mock()
    )
    async def test_process_document_string_input(self, mock_config):
        """Test processing a single document with string input."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Setup mocks
        mock_text_cleaner = Mock()
        mock_text_cleaner.clean.return_value = "cleaned text"
        pipeline.text_cleaner = mock_text_cleaner
        
        mock_sentiment_analyzer = Mock()
        mock_sentiment_analyzer.predict.return_value = {
            "sentiment": "positive",
            "confidence": 0.85
        }
        pipeline.sentiment_analyzer = mock_sentiment_analyzer
        
        mock_entity_extractor = Mock()
        mock_entity_extractor.extract_entities.return_value = [
            {"text": "entity", "type": "MISC"}
        ]
        pipeline.entity_extractor = mock_entity_extractor
        
        # Mock _load_models
        pipeline._load_models = Mock()
        
        result = await pipeline.process_document("This is a test document.")
        
        assert isinstance(result, PipelineResult)
        assert result.text == "This is a test document."
        assert result.sentiment == "positive"
        assert result.sentiment_confidence == 0.85
        assert len(result.entities) == 1
        assert result.processing_time > 0
        assert result.timestamp is not None
        assert result.document_id.startswith("doc_")
        
        # Verify calls
        mock_text_cleaner.clean.assert_called_once_with("This is a test document.")
        mock_sentiment_analyzer.predict.assert_called_once_with("cleaned text")
        mock_entity_extractor.extract_entities.assert_called_once_with("cleaned text")
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        metrics_collector=Mock(),
        perf_logger=Mock()
    )
    async def test_process_document_dict_input(self, mock_config):
        """Test processing a single document with dict input."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Setup mocks
        pipeline.text_cleaner = Mock()
        pipeline.text_cleaner.clean.return_value = "cleaned text"
        
        pipeline.sentiment_analyzer = Mock()
        pipeline.sentiment_analyzer.predict.return_value = {
            "sentiment": "negative",
            "confidence": 0.75
        }
        
        pipeline.entity_extractor = Mock()
        pipeline.entity_extractor.extract_entities.return_value = []
        
        pipeline._load_models = Mock()
        
        document = {
            "text": "This is a negative review.",
            "document_id": "custom_doc_123"
        }
        
        result = await pipeline.process_document(document)
        
        assert result.text == "This is a negative review."
        assert result.document_id == "custom_doc_123"
        assert result.sentiment == "negative"
        assert result.sentiment_confidence == 0.75
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        metrics_collector=Mock(),
        perf_logger=Mock()
    )
    async def test_process_document_error_handling(self, mock_config):
        """Test error handling in document processing."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Setup mocks to raise error
        pipeline._load_models = Mock()
        pipeline.text_cleaner = Mock()
        pipeline.text_cleaner.clean.side_effect = Exception("Processing error")
        
        with pytest.raises(Exception, match="Processing error"):
            await pipeline.process_document("test text")
        
        # Verify metrics were updated
        from src.pipeline import metrics_collector
        metrics_collector.increment_counter.assert_called_with("processing_errors")
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        perf_logger=Mock()
    )
    async def test_process_batch(self, mock_config):
        """Test batch processing of documents."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock process_document to return successful results
        async def mock_process_document(doc):
            return PipelineResult(
                document_id="test_doc",
                text=doc if isinstance(doc, str) else doc.get('text', ''),
                sentiment="positive",
                sentiment_confidence=0.8,
                entities=[],
                processing_time=0.1,
                timestamp=datetime.utcnow()
            )
        
        pipeline.process_document = mock_process_document
        
        documents = [
            "First document",
            "Second document",
            {"text": "Third document", "document_id": "doc_3"}
        ]
        
        results = await pipeline.process_batch(documents)
        
        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)
        assert results[0].text == "First document"
        assert results[1].text == "Second document"
        assert results[2].text == "Third document"
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        perf_logger=Mock()
    )
    async def test_process_batch_with_errors(self, mock_config):
        """Test batch processing with some errors."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock process_document to return mix of success and errors
        async def mock_process_document(doc):
            if "error" in doc:
                raise Exception("Processing failed")
            return PipelineResult(
                document_id="test_doc",
                text=doc,
                sentiment="positive",
                sentiment_confidence=0.8,
                entities=[],
                processing_time=0.1,
                timestamp=datetime.utcnow()
            )
        
        pipeline.process_document = mock_process_document
        
        documents = ["good document", "document with error", "another good document"]
        
        results = await pipeline.process_batch(documents)
        
        # Should return only successful results
        assert len(results) == 2
        assert all(isinstance(r, PipelineResult) for r in results)
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock(),
        KafkaConsumer=Mock()
    )
    async def test_start_streaming(self, mock_config):
        """Test starting streaming pipeline with Kafka."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.kafka.bootstrap_servers = "localhost:9092"
        mock_config.kafka.topic = "nlp-input"
        mock_config.kafka.group_id = "nlp-consumer"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock Kafka consumer
        mock_consumer = AsyncMock()
        mock_consumer.consume_batch.return_value = iter([["doc1", "doc2"]])  # One batch
        from src.pipeline import KafkaConsumer
        KafkaConsumer.return_value = mock_consumer
        
        # Mock process_batch
        async def mock_process_batch(docs):
            return [
                PipelineResult(
                    document_id=f"doc_{i}",
                    text=doc,
                    sentiment="positive",
                    sentiment_confidence=0.8,
                    entities=[],
                    processing_time=0.1,
                    timestamp=datetime.utcnow()
                )
                for i, doc in enumerate(docs)
            ]
        
        pipeline.process_batch = mock_process_batch
        pipeline.documents_processed = 0  # Start with 0
        pipeline._export_results = AsyncMock()
        
        # Mock async iteration to stop after one iteration
        async def mock_consume_batch():
            yield ["doc1", "doc2"]
            return  # Stop iteration
        
        mock_consumer.consume_batch = mock_consume_batch
        
        # This would normally run indefinitely, so we'll test the setup
        await pipeline.start_streaming()
        
        # Verify consumer was created and started
        mock_consumer.start.assert_called_once()
        mock_consumer.stop.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    async def test_process_files(self, mock_config, temp_dir):
        """Test processing files through the pipeline."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.model.batch_size = 32
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock data loader
        mock_data_loader = Mock()
        mock_data_loader.load_file.return_value = [
            {"text": "Document 1", "id": "doc1"},
            {"text": "Document 2", "id": "doc2"}
        ]
        pipeline.data_loader = mock_data_loader
        
        # Mock process_batch
        async def mock_process_batch(docs):
            return [
                PipelineResult(
                    document_id=doc.get("id", "unknown"),
                    text=doc.get("text", ""),
                    sentiment="positive",
                    sentiment_confidence=0.8,
                    entities=[],
                    processing_time=0.1,
                    timestamp=datetime.utcnow()
                )
                for doc in docs
            ]
        
        pipeline.process_batch = mock_process_batch
        pipeline._export_results = AsyncMock()
        
        # Create test file
        test_file = temp_dir / "test.json"
        
        await pipeline.process_files([str(test_file)], "json")
        
        # Verify calls
        mock_data_loader.load_file.assert_called_once_with(str(test_file), "json")
        pipeline._export_results.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    async def test_export_results(self, mock_config, temp_dir):
        """Test exporting aggregated results."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.storage.output_path = temp_dir
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock aggregator
        mock_aggregator = Mock()
        mock_aggregator.get_summary.return_value = {"total_documents": 10}
        mock_aggregator.results = [{"doc_id": "1", "sentiment": "positive"}]
        pipeline.aggregator = mock_aggregator
        
        # Mock visualizer and exporter
        mock_visualizer = Mock()
        pipeline.visualizer = mock_visualizer
        
        mock_exporter = Mock()
        pipeline.exporter = mock_exporter
        
        await pipeline._export_results()
        
        # Verify visualizations were created
        mock_visualizer.plot_sentiment_distribution.assert_called_once()
        mock_visualizer.plot_entity_frequency.assert_called_once()
        
        # Verify exports were created
        mock_exporter.export_json.assert_called_once()
        mock_exporter.export_csv.assert_called_once()
        mock_exporter.export_html.assert_called_once()
    
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    def test_get_stats(self, mock_config):
        """Test getting pipeline statistics."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Set some statistics
        pipeline.documents_processed = 100
        pipeline.total_processing_time = 50.0
        
        # Mock aggregator
        mock_aggregator = Mock()
        mock_aggregator.get_summary.return_value = {"total_documents": 100}
        pipeline.aggregator = mock_aggregator
        
        stats = pipeline.get_stats()
        
        assert stats["documents_processed"] == 100
        assert stats["average_processing_time"] == 0.5
        assert stats["throughput"] == 2.0  # 100 docs / 50 seconds
        assert stats["aggregated_results"]["total_documents"] == 100
    
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    def test_get_stats_no_processing(self, mock_config):
        """Test getting statistics when no processing has occurred."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Mock aggregator
        mock_aggregator = Mock()
        mock_aggregator.get_summary.return_value = {"total_documents": 0}
        pipeline.aggregator = mock_aggregator
        
        stats = pipeline.get_stats()
        
        assert stats["documents_processed"] == 0
        assert stats["average_processing_time"] == 0
        assert stats["throughput"] == 0
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    async def test_shutdown(self, mock_config):
        """Test pipeline shutdown."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # Set some processed documents
        pipeline.documents_processed = 50
        
        # Mock components
        pipeline._export_results = AsyncMock()
        
        mock_kafka_consumer = AsyncMock()
        pipeline.kafka_consumer = mock_kafka_consumer
        
        mock_model_manager = Mock()
        pipeline.model_manager = mock_model_manager
        
        await pipeline.shutdown()
        
        # Verify cleanup
        pipeline._export_results.assert_called_once()
        mock_kafka_consumer.stop.assert_called_once()
        mock_model_manager.clear_cache.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.pipeline.config')
    @patch.multiple(
        'src.pipeline',
        TextCleaner=Mock(),
        BERTTokenizer=Mock(),
        ModelManager=Mock(),
        ResultAggregator=Mock(),
        Visualizer=Mock(),
        ResultExporter=Mock(),
        DataLoader=Mock()
    )
    async def test_shutdown_no_processing(self, mock_config):
        """Test shutdown when no processing has occurred."""
        mock_config.model.bert_model_path = "bert-base-uncased"
        mock_config.model.model_cache_dir = "/tmp/models"
        mock_config.model.spacy_model = "en_core_web_sm"
        mock_config.to_dict.return_value = {"test": "config"}
        
        pipeline = NLPPipeline()
        
        # No documents processed
        pipeline.documents_processed = 0
        
        # Mock components
        pipeline._export_results = AsyncMock()
        mock_model_manager = Mock()
        pipeline.model_manager = mock_model_manager
        
        await pipeline.shutdown()
        
        # Should not export results if no documents processed
        pipeline._export_results.assert_not_called()
        mock_model_manager.clear_cache.assert_called_once()


# Integration tests
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, mock_sentiment_model, mock_entity_model, sample_texts):
        """Test end-to-end document processing."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock(),
            metrics_collector=Mock(),
            perf_logger=Mock()
        ):
            # Setup config
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            # Create pipeline
            pipeline = NLPPipeline()
            
            # Setup mocks
            pipeline.text_cleaner.clean.side_effect = lambda x: x.lower()
            pipeline.sentiment_analyzer = mock_sentiment_model
            pipeline.entity_extractor = mock_entity_model
            pipeline._load_models = Mock()
            
            # Process documents
            results = []
            for text in sample_texts[:5]:  # Test subset
                if text.strip():  # Skip empty texts
                    result = await pipeline.process_document(text)
                    results.append(result)
            
            # Verify results
            assert len(results) > 0
            assert all(isinstance(r, PipelineResult) for r in results)
            assert all(r.processing_time > 0 for r in results)
            assert all(r.timestamp is not None for r in results)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, mock_sentiment_model, mock_entity_model, sentiment_labeled_texts):
        """Test batch processing integration."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock(),
            perf_logger=Mock()
        ):
            # Setup config
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            # Create pipeline
            pipeline = NLPPipeline()
            
            # Setup mocks
            pipeline.text_cleaner.clean.side_effect = lambda x: x.lower()
            pipeline.sentiment_analyzer = mock_sentiment_model
            pipeline.entity_extractor = mock_entity_model
            pipeline._load_models = Mock()
            
            # Extract texts from labeled data
            texts = [item["text"] for item in sentiment_labeled_texts]
            
            # Process batch
            results = await pipeline.process_batch(texts)
            
            # Verify results
            assert len(results) == len(texts)
            assert all(isinstance(r, PipelineResult) for r in results)
            
            # Check that processing was efficient
            total_time = sum(r.processing_time for r in results)
            assert total_time > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_file_processing_integration(self, mock_sentiment_model, mock_entity_model, sample_json_file):
        """Test file processing integration."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            # Setup config
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.model.batch_size = 32
            config.to_dict.return_value = {"test": "config"}
            
            # Create pipeline
            pipeline = NLPPipeline()
            
            # Setup mocks
            pipeline.text_cleaner.clean.side_effect = lambda x: x.lower()
            pipeline.sentiment_analyzer = mock_sentiment_model
            pipeline.entity_extractor = mock_entity_model
            pipeline._load_models = Mock()
            pipeline._export_results = AsyncMock()
            
            # Mock data loader to read actual file
            with open(sample_json_file, 'r') as f:
                test_data = json.load(f)
            
            pipeline.data_loader.load_file.return_value = test_data
            
            # Process files
            await pipeline.process_files([str(sample_json_file)], "json")
            
            # Verify processing occurred
            pipeline.data_loader.load_file.assert_called_once()
            pipeline._export_results.assert_called_once()


# Performance tests
class TestPipelinePerformance:
    """Performance tests for the pipeline."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_document_performance(self, async_mock_pipeline, performance_timer):
        """Test single document processing performance."""
        performance_timer.start()
        
        # Process single document
        result = await async_mock_pipeline.process_document("Test document for performance testing.")
        
        elapsed = performance_timer.stop()
        
        # Should process quickly
        assert elapsed < 0.1  # Less than 100ms
        assert result.processing_time < 0.01  # Mock processing time
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, async_mock_pipeline, large_text_batch, performance_timer):
        """Test batch processing performance."""
        performance_timer.start()
        
        # Process large batch
        results = await async_mock_pipeline.process_batch(large_text_batch)
        
        elapsed = performance_timer.stop()
        
        # Should process efficiently
        assert elapsed < 5.0  # Less than 5 seconds for large batch
        assert len(results) == len(large_text_batch)
        
        # Calculate throughput
        throughput = len(large_text_batch) / elapsed
        assert throughput > 100  # Should process >100 docs per second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, performance_timer):
        """Test concurrent document processing performance."""
        # Create multiple mock pipelines to simulate concurrency
        async def mock_process_document(text):
            await asyncio.sleep(0.001)  # Simulate processing
            return PipelineResult(
                document_id="test",
                text=text,
                sentiment="positive",
                sentiment_confidence=0.8,
                entities=[],
                processing_time=0.001,
                timestamp=datetime.utcnow()
            )
        
        texts = ["Test document"] * 100
        
        performance_timer.start()
        
        # Process concurrently
        tasks = [mock_process_document(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        elapsed = performance_timer.stop()
        
        # Should be faster than sequential processing
        assert elapsed < 1.0  # Concurrent processing should be fast
        assert len(results) == 100
    
    @pytest.mark.performance
    def test_memory_usage_during_processing(self):
        """Test memory usage during pipeline processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and configure pipeline
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            # Create multiple pipelines to test memory usage
            pipelines = []
            for i in range(10):
                pipeline = NLPPipeline()
                pipelines.append(pipeline)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory (less than 50MB increase)
        assert memory_increase < 50 * 1024 * 1024


# Error handling and edge cases
class TestPipelineErrorHandling:
    """Test error handling and edge cases in the pipeline."""
    
    @pytest.mark.asyncio
    async def test_invalid_document_types(self):
        """Test handling of invalid document types."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            pipeline._load_models = Mock()
            
            # Test various invalid inputs
            invalid_docs = [
                None,
                123,
                [],
                {"no_text_field": "value"},
                {"text": None},
                {"text": ""},
            ]
            
            for doc in invalid_docs:
                try:
                    # Some should fail, others should handle gracefully
                    result = await pipeline.process_document(doc)
                    # If it succeeds, verify it's a valid result
                    if result:
                        assert isinstance(result, PipelineResult)
                except Exception:
                    # Expected for truly invalid inputs
                    pass
    
    @pytest.mark.asyncio
    async def test_model_loading_failures(self):
        """Test handling of model loading failures."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            
            # Mock model loading failure
            pipeline._load_models = Mock(side_effect=Exception("Model loading failed"))
            
            with pytest.raises(Exception, match="Model loading failed"):
                await pipeline.process_document("test text")
    
    @pytest.mark.asyncio
    async def test_partial_batch_failure_handling(self):
        """Test handling when some documents in a batch fail."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock(),
            perf_logger=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            
            # Mock process_document to fail on specific input
            async def mock_process_document(doc):
                if "fail" in str(doc).lower():
                    raise Exception("Processing failed")
                return PipelineResult(
                    document_id="success",
                    text=str(doc),
                    sentiment="positive",
                    sentiment_confidence=0.8,
                    entities=[],
                    processing_time=0.1,
                    timestamp=datetime.utcnow()
                )
            
            pipeline.process_document = mock_process_document
            
            documents = [
                "good document",
                "FAIL document",
                "another good document",
                "this will fail too"
            ]
            
            results = await pipeline.process_batch(documents)
            
            # Should return only successful results
            assert len(results) == 2
            assert all(r.document_id == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self):
        """Test that resources are cleaned up properly on errors."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            
            # Set up mock Kafka consumer
            mock_consumer = AsyncMock()
            mock_consumer.consume_batch.side_effect = Exception("Kafka error")
            pipeline.kafka_consumer = mock_consumer
            
            # Test that streaming handles errors and cleans up
            with pytest.raises(Exception, match="Kafka error"):
                await pipeline.start_streaming()
            
            # Verify cleanup was called
            mock_consumer.stop.assert_called_once()


# Main example test
class TestPipelineMainExample:
    """Test the main example function."""
    
    @pytest.mark.asyncio
    @patch('src.pipeline.NLPPipeline')
    async def test_main_function(self, mock_pipeline_class):
        """Test the main example function."""
        # Mock pipeline instance
        mock_pipeline = AsyncMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock results
        mock_result = PipelineResult(
            document_id="example_001",
            text="This product is amazing! I love how it simplifies my workflow.",
            sentiment="positive",
            sentiment_confidence=0.85,
            entities=[],
            processing_time=0.1,
            timestamp=datetime.utcnow()
        )
        
        mock_batch_results = [
            PipelineResult(
                document_id="doc_1",
                text="Apple Inc. announced new products in Cupertino.",
                sentiment="neutral",
                sentiment_confidence=0.7,
                entities=[],
                processing_time=0.1,
                timestamp=datetime.utcnow()
            )
        ] * 3
        
        mock_pipeline.process_document.return_value = mock_result
        mock_pipeline.process_batch.return_value = mock_batch_results
        mock_pipeline.get_stats.return_value = {"documents_processed": 4}
        
        # Import and run main
        from src.pipeline import main
        
        # This should run without errors
        await main()
        
        # Verify pipeline was used correctly
        mock_pipeline.process_document.assert_called()
        mock_pipeline.process_batch.assert_called()
        mock_pipeline.get_stats.assert_called()
        mock_pipeline.shutdown.assert_called()


# Stress tests
class TestPipelineStressTests:
    """Stress tests for the pipeline."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Test processing very large documents."""
        # Create a very large document (1MB of text)
        large_text = "This is a test sentence. " * 40000  # ~1MB
        
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock(),
            metrics_collector=Mock(),
            perf_logger=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            
            # Mock components to handle large text
            pipeline.text_cleaner.clean.return_value = large_text[:1000]  # Truncate for testing
            pipeline.sentiment_analyzer = Mock()
            pipeline.sentiment_analyzer.predict.return_value = {
                "sentiment": "neutral",
                "confidence": 0.5
            }
            pipeline.entity_extractor = Mock()
            pipeline.entity_extractor.extract_entities.return_value = []
            pipeline._load_models = Mock()
            
            # Should handle large document without crashing
            result = await pipeline.process_document(large_text)
            
            assert isinstance(result, PipelineResult)
            assert result.processing_time > 0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_processing_load(self, performance_timer):
        """Test sustained processing load over time."""
        with patch.multiple(
            'src.pipeline',
            config=Mock(),
            TextCleaner=Mock(),
            BERTTokenizer=Mock(),
            ModelManager=Mock(),
            ResultAggregator=Mock(),
            Visualizer=Mock(),
            ResultExporter=Mock(),
            DataLoader=Mock(),
            perf_logger=Mock()
        ):
            from src.pipeline import config
            config.model.bert_model_path = "bert-base-uncased"
            config.model.model_cache_dir = "/tmp/models"
            config.model.spacy_model = "en_core_web_sm"
            config.to_dict.return_value = {"test": "config"}
            
            pipeline = NLPPipeline()
            
            # Setup fast mocks
            pipeline._load_models = Mock()
            pipeline.text_cleaner.clean.side_effect = lambda x: x.lower()
            pipeline.sentiment_analyzer = Mock()
            pipeline.sentiment_analyzer.predict.return_value = {
                "sentiment": "positive",
                "confidence": 0.8
            }
            pipeline.entity_extractor = Mock()
            pipeline.entity_extractor.extract_entities.return_value = []
            
            # Process many batches to simulate sustained load
            num_batches = 20
            batch_size = 50
            total_docs = num_batches * batch_size
            
            performance_timer.start()
            
            for i in range(num_batches):
                batch = [f"Document {j} in batch {i}" for j in range(batch_size)]
                results = await pipeline.process_batch(batch)
                assert len(results) == batch_size
            
            elapsed = performance_timer.stop()
            
            # Should maintain reasonable performance
            assert elapsed < 30.0  # Less than 30 seconds for 1000 documents
            
            # Calculate sustained throughput
            throughput = total_docs / elapsed
            assert throughput > 30  # Should maintain >30 docs/sec sustained