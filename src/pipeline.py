"""Main NLP Pipeline orchestrator."""

import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
from dataclasses import dataclass

try:
    from src.ingestion import KafkaConsumer, SparkProcessor, DataLoader
except ImportError:
    # Make Kafka/Spark optional
    KafkaConsumer = None
    SparkProcessor = None
    from src.ingestion.data_loader import DataLoader
from src.preprocessing import TextCleaner, BERTTokenizer
from src.models import SentimentAnalyzer, EntityExtractor, ModelManager
from src.postprocessing import ResultAggregator, Visualizer, ResultExporter
from src.monitoring import metrics_collector, health_checker
from src.utils.config import config
from src.utils.logger import get_logger, perf_logger


logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    document_id: str
    text: str
    sentiment: str
    sentiment_confidence: float
    entities: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime


class NLPPipeline:
    """Main NLP Pipeline orchestrator."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the NLP Pipeline."""
        self.config = config
        if config_file:
            self.config.load_from_file(config_file)
        
        # Initialize components
        self._init_components()
        
        # Metrics
        self.documents_processed = 0
        self.total_processing_time = 0.0
        
        logger.info("NLP Pipeline initialized", extra={"config": self.config.to_dict()})
    
    def _init_components(self):
        """Initialize pipeline components."""
        # Preprocessing
        self.text_cleaner = TextCleaner()
        self.tokenizer = BERTTokenizer(model_name=self.config.model.bert_model_path)
        
        # Models
        self.model_manager = ModelManager(cache_dir=self.config.model.model_cache_dir)
        self.sentiment_analyzer = None  # Lazy load
        self.entity_extractor = None   # Lazy load
        
        # Postprocessing
        self.aggregator = ResultAggregator()
        self.visualizer = Visualizer()
        self.exporter = ResultExporter(output_dir=str(self.config.storage.output_path))
        
        # Data ingestion
        self.kafka_consumer = None
        self.spark_processor = None
        self.data_loader = DataLoader()
    
    def _load_models(self):
        """Lazy load models."""
        if not self.sentiment_analyzer:
            # Use a smaller model for faster testing
            model_name = "distilbert-base-uncased-finetuned-sst-2-english" if "bert-base" in self.config.model.bert_model_path else self.config.model.bert_model_path
            self.sentiment_analyzer = self.model_manager.load_model(
                "sentiment", 
                model_name=model_name
            )
        
        if not self.entity_extractor:
            self.entity_extractor = self.model_manager.load_model(
                "entity",
                model_name=self.config.model.spacy_model
            )
    
    async def process_document(self, document: Union[str, Dict[str, Any]]) -> PipelineResult:
        """
        Process a single document through the pipeline.
        
        Args:
            document: Text string or dict with 'text' and optional 'document_id'
        
        Returns:
            PipelineResult with analysis results
        """
        start_time = time.time()
        
        # Extract text and ID
        if isinstance(document, str):
            text = document
            doc_id = f"doc_{int(time.time() * 1000)}"
        else:
            text = document.get('text', '')
            doc_id = document.get('document_id', f"doc_{int(time.time() * 1000)}")
        
        try:
            # Load models if needed
            self._load_models()
            
            # Preprocessing
            clean_text = self.text_cleaner.clean(text)
            
            # Sentiment analysis
            sentiment_result = await asyncio.to_thread(
                self.sentiment_analyzer.predict, clean_text
            )
            
            # Entity extraction
            entity_result = await asyncio.to_thread(
                self.entity_extractor.predict, clean_text
            )
            entities = entity_result.get('entities', [])
            
            # Create result
            processing_time = time.time() - start_time
            result = PipelineResult(
                document_id=doc_id,
                text=text,
                sentiment=sentiment_result['sentiment'],
                sentiment_confidence=sentiment_result['confidence'],
                entities=entities,
                processing_time=processing_time,
                timestamp=datetime.utcnow()
            )
            
            # Update metrics
            self.documents_processed += 1
            self.total_processing_time += processing_time
            metrics_collector.increment_counter("documents_processed")
            metrics_collector.record_histogram("processing_time", processing_time)
            
            # Log performance
            perf_logger.log_timing("document_processing", processing_time,
                                 document_id=doc_id,
                                 text_length=len(text))
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}",
                        extra={"document_id": doc_id, "error": str(e)})
            metrics_collector.increment_counter("processing_errors")
            raise
    
    async def process_batch(self, documents: List[Union[str, Dict[str, Any]]]) -> List[PipelineResult]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of documents to process
        
        Returns:
            List of PipelineResult objects
        """
        logger.info(f"Processing batch of {len(documents)} documents")
        start_time = time.time()
        
        # Process documents concurrently
        tasks = [self.process_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        successful_results = [r for r in results if isinstance(r, PipelineResult)]
        failed_count = len(results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} documents failed processing")
        
        # Log batch performance
        batch_time = time.time() - start_time
        perf_logger.log_throughput("batch_processing", len(successful_results), 
                                 batch_time, batch_size=len(documents))
        
        return successful_results
    
    async def start_streaming(self):
        """Start streaming pipeline with Kafka."""
        if KafkaConsumer is None:
            raise RuntimeError("Kafka dependencies not installed. Install with: pip install confluent-kafka")
            
        logger.info("Starting streaming pipeline")
        
        # Initialize Kafka consumer
        self.kafka_consumer = KafkaConsumer(
            bootstrap_servers=self.config.kafka.bootstrap_servers,
            topic=self.config.kafka.topic,
            group_id=self.config.kafka.group_id
        )
        
        # Start consuming messages
        await self.kafka_consumer.start()
        
        try:
            async for batch in self.kafka_consumer.consume_batch():
                # Process batch
                results = await self.process_batch(batch)
                
                # Aggregate results
                self.aggregator.add_results([
                    {
                        'document_id': r.document_id,
                        'text': r.text,
                        'sentiment': r.sentiment,
                        'confidence': r.sentiment_confidence,
                        'entities': r.entities,
                        'timestamp': r.timestamp,
                        'processing_time': r.processing_time
                    }
                    for r in results
                ])
                
                # Export results periodically
                if self.documents_processed % 1000 == 0:
                    await self._export_results()
        
        finally:
            await self.kafka_consumer.stop()
    
    async def process_files(self, file_paths: List[str], file_format: str = "json"):
        """
        Process files through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            file_format: Format of the files
        """
        logger.info(f"Processing {len(file_paths)} files")
        
        for file_path in file_paths:
            # Load documents from file
            documents = list(self.data_loader.load_file(file_path, file_format))
            
            # Process in batches
            batch_size = self.config.model.batch_size
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                results = await self.process_batch(batch)
                
                # Add to aggregator
                self.aggregator.add_results([
                    {
                        'document_id': r.document_id,
                        'text': r.text,
                        'sentiment': r.sentiment,
                        'confidence': r.sentiment_confidence,
                        'entities': r.entities,
                        'timestamp': r.timestamp,
                        'processing_time': r.processing_time
                    }
                    for r in results
                ])
        
        # Export final results
        await self._export_results()
    
    async def _export_results(self):
        """Export aggregated results."""
        logger.info("Exporting results")
        
        # Get aggregated results
        aggregated = self.aggregator.get_summary_statistics()
        
        # Generate visualizations
        if aggregated['total_documents'] > 0:
            # Create sentiment distribution chart
            self.visualizer.plot_sentiment_distribution(
                self.aggregator.results,
                save_path=self.config.storage.output_path / "sentiment_distribution.html"
            )
            
            # Create entity frequency chart
            self.visualizer.plot_entity_frequency(
                self.aggregator.results,
                save_path=self.config.storage.output_path / "entity_frequency.html"
            )
        
        # Export to multiple formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON export
        self.exporter.export_to_json(
            self.aggregator.results,
            f"results_{timestamp}.json"
        )
        
        # CSV export
        self.exporter.export_to_csv(
            self.aggregator.results,
            f"results_{timestamp}.csv"
        )
        
        # HTML report
        self.exporter.export_to_html(
            self.aggregator.results,
            report_data={"title": "NLP Pipeline Analysis Report", "stats": aggregated},
            filename=f"report_{timestamp}.html"
        )
        
        logger.info(f"Results exported to {self.config.storage.output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_processing_time = (
            self.total_processing_time / self.documents_processed 
            if self.documents_processed > 0 else 0
        )
        
        return {
            "documents_processed": self.documents_processed,
            "average_processing_time": avg_processing_time,
            "throughput": self.documents_processed / self.total_processing_time if self.total_processing_time > 0 else 0,
            "aggregated_results": self.aggregator.get_summary_statistics()
        }
    
    async def shutdown(self):
        """Shutdown pipeline gracefully."""
        logger.info("Shutting down NLP Pipeline")
        
        # Export final results
        if self.documents_processed > 0:
            await self._export_results()
        
        # Clean up resources
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        
        # Clear model cache
        self.model_manager.unload_all()
        
        logger.info("NLP Pipeline shutdown complete")


# Example usage
async def main():
    """Example pipeline usage."""
    # Initialize pipeline
    pipeline = NLPPipeline()
    
    # Example 1: Process single document
    result = await pipeline.process_document({
        "text": "This product is amazing! I love how it simplifies my workflow.",
        "document_id": "example_001"
    })
    print(f"Sentiment: {result.sentiment} (confidence: {result.sentiment_confidence:.2f})")
    print(f"Entities: {result.entities}")
    
    # Example 2: Process batch
    documents = [
        "Apple Inc. announced new products in Cupertino.",
        "The customer service was terrible and unhelpful.",
        "Microsoft Azure provides excellent cloud services."
    ]
    results = await pipeline.process_batch(documents)
    for r in results:
        print(f"Doc {r.document_id}: {r.sentiment} ({r.sentiment_confidence:.2f})")
    
    # Example 3: Process files
    # await pipeline.process_files(["data/documents.json"])
    
    # Get statistics
    stats = pipeline.get_stats()
    print(f"Pipeline stats: {stats}")
    
    # Shutdown
    await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())