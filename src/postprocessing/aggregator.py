"""Aggregate NLP processing results across multiple documents."""

from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AggregationMetrics:
    """Container for aggregation metrics."""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    sentiment_confidence_avg: float = 0.0
    sentiment_confidence_std: float = 0.0
    
    total_entities: int = 0
    unique_entities: int = 0
    entity_type_distribution: Dict[str, int] = field(default_factory=dict)
    entities_per_document_avg: float = 0.0
    
    processing_time_avg: float = 0.0
    text_length_avg: float = 0.0
    word_count_avg: float = 0.0


class ResultAggregator:
    """Aggregate and analyze results across multiple documents."""
    
    def __init__(self):
        """Initialize result aggregator."""
        self.results = []
        self.metrics = AggregationMetrics()
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Add results to the aggregator.
        
        Args:
            results: List of document processing results
        """
        self.results.extend(results)
        self._update_metrics()
    
    def aggregate_by_sentiment(self) -> Dict[str, Any]:
        """
        Aggregate results by sentiment.
        
        Returns:
            Dictionary with sentiment-based aggregations
        """
        sentiment_groups = defaultdict(list)
        
        for result in self.results:
            if "sentiment" in result and "label" in result["sentiment"]:
                sentiment = result["sentiment"]["label"]
                sentiment_groups[sentiment].append(result)
        
        aggregated = {}
        for sentiment, group in sentiment_groups.items():
            aggregated[sentiment] = {
                "count": len(group),
                "percentage": len(group) / len(self.results) * 100,
                "average_confidence": self._calculate_avg_confidence(group),
                "common_entities": self._get_common_entities(group, top_n=10),
                "text_statistics": self._calculate_text_stats(group)
            }
        
        return aggregated
    
    def aggregate_by_entity_type(self) -> Dict[str, Any]:
        """
        Aggregate results by entity types.
        
        Returns:
            Dictionary with entity-type-based aggregations
        """
        entity_type_groups = defaultdict(list)
        
        for result in self.results:
            if "entities" in result:
                for entity in result["entities"]:
                    entity_type = entity.get("label", "UNKNOWN")
                    entity_type_groups[entity_type].append({
                        "text": entity.get("text", ""),
                        "score": entity.get("score", 0.0),
                        "document_id": result.get("document_id", "unknown")
                    })
        
        aggregated = {}
        for entity_type, entities in entity_type_groups.items():
            unique_texts = list(set(e["text"] for e in entities))
            aggregated[entity_type] = {
                "total_occurrences": len(entities),
                "unique_entities": len(unique_texts),
                "average_confidence": statistics.mean([e["score"] for e in entities]),
                "top_entities": self._get_top_entities_by_type(entities, top_n=20),
                "document_coverage": len(set(e["document_id"] for e in entities)) / len(self.results)
            }
        
        return aggregated
    
    def aggregate_by_time_window(self, window_size: timedelta = timedelta(days=1),
                               date_field: str = "processed_at") -> Dict[str, Any]:
        """
        Aggregate results by time windows.
        
        Args:
            window_size: Size of time window for aggregation
            date_field: Field containing timestamp
            
        Returns:
            Dictionary with time-based aggregations
        """
        time_groups = defaultdict(list)
        
        for result in self.results:
            if date_field in result:
                try:
                    timestamp = datetime.fromisoformat(result[date_field])
                    window_key = timestamp.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    time_groups[window_key].append(result)
                except:
                    continue
        
        aggregated = {}
        for window, group in sorted(time_groups.items()):
            aggregated[window.isoformat()] = {
                "document_count": len(group),
                "sentiment_distribution": self._get_sentiment_distribution(group),
                "entity_count": sum(len(r.get("entities", [])) for r in group),
                "unique_entities": len(self._get_unique_entities(group)),
                "average_confidence": self._calculate_avg_confidence(group)
            }
        
        return aggregated
    
    def aggregate_by_classification(self) -> Dict[str, Any]:
        """
        Aggregate results by classification categories.
        
        Returns:
            Dictionary with classification-based aggregations
        """
        classification_groups = defaultdict(list)
        
        for result in self.results:
            if "classification" in result and "label" in result["classification"]:
                category = result["classification"]["label"]
                classification_groups[category].append(result)
        
        aggregated = {}
        for category, group in classification_groups.items():
            aggregated[category] = {
                "count": len(group),
                "percentage": len(group) / len(self.results) * 100,
                "sentiment_distribution": self._get_sentiment_distribution(group),
                "common_entities": self._get_common_entities(group, top_n=10),
                "average_classification_confidence": statistics.mean(
                    [r["classification"].get("score", 0) for r in group]
                )
            }
        
        return aggregated
    
    def get_entity_relationships(self, min_co_occurrence: int = 3) -> List[Dict[str, Any]]:
        """
        Find relationships between entities based on co-occurrence.
        
        Args:
            min_co_occurrence: Minimum times entities must appear together
            
        Returns:
            List of entity relationships
        """
        co_occurrences = Counter()
        
        for result in self.results:
            if "entities" in result:
                entities = list(set(e.get("text", "") for e in result["entities"]))
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_occurrences[pair] += 1
        
        relationships = []
        for (entity1, entity2), count in co_occurrences.items():
            if count >= min_co_occurrence:
                relationships.append({
                    "entity1": entity1,
                    "entity2": entity2,
                    "co_occurrence_count": count,
                    "strength": count / len(self.results)
                })
        
        return sorted(relationships, key=lambda x: x["co_occurrence_count"], reverse=True)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "total_documents": self.metrics.total_documents,
            "success_rate": self.metrics.successful_documents / self.metrics.total_documents if self.metrics.total_documents > 0 else 0,
            "sentiment": {
                "distribution": self.metrics.sentiment_distribution,
                "average_confidence": self.metrics.sentiment_confidence_avg,
                "confidence_std_dev": self.metrics.sentiment_confidence_std
            },
            "entities": {
                "total": self.metrics.total_entities,
                "unique": self.metrics.unique_entities,
                "average_per_document": self.metrics.entities_per_document_avg,
                "type_distribution": self.metrics.entity_type_distribution
            },
            "text": {
                "average_length": self.metrics.text_length_avg,
                "average_words": self.metrics.word_count_avg
            },
            "performance": {
                "average_processing_time": self.metrics.processing_time_avg
            }
        }
    
    def find_outliers(self, metric: str = "sentiment_confidence", 
                     threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Find outlier documents based on specified metric.
        
        Args:
            metric: Metric to use for outlier detection
            threshold: Number of standard deviations for outlier threshold
            
        Returns:
            List of outlier documents
        """
        values = []
        
        if metric == "sentiment_confidence":
            values = [(i, r["sentiment"]["score"]) 
                     for i, r in enumerate(self.results) 
                     if "sentiment" in r and "score" in r["sentiment"]]
        elif metric == "entity_count":
            values = [(i, len(r.get("entities", []))) 
                     for i, r in enumerate(self.results)]
        elif metric == "text_length":
            values = [(i, len(r.get("text", ""))) 
                     for i, r in enumerate(self.results)]
        
        if not values:
            return []
        
        indices, scores = zip(*values)
        mean_val = statistics.mean(scores)
        std_val = statistics.stdev(scores) if len(scores) > 1 else 0
        
        outliers = []
        for idx, score in zip(indices, scores):
            z_score = abs(score - mean_val) / std_val if std_val > 0 else 0
            if z_score > threshold:
                outliers.append({
                    "document_id": self.results[idx].get("document_id", f"doc_{idx}"),
                    "metric": metric,
                    "value": score,
                    "z_score": z_score,
                    "mean": mean_val,
                    "std_dev": std_val
                })
        
        return outliers
    
    def get_trends(self, date_field: str = "processed_at") -> Dict[str, Any]:
        """
        Analyze trends over time.
        
        Args:
            date_field: Field containing timestamp
            
        Returns:
            Dictionary with trend analysis
        """
        time_series = defaultdict(lambda: {
            "sentiment": defaultdict(int),
            "entities": 0,
            "documents": 0
        })
        
        for result in self.results:
            if date_field in result:
                try:
                    timestamp = datetime.fromisoformat(result[date_field])
                    date_key = timestamp.date().isoformat()
                    
                    time_series[date_key]["documents"] += 1
                    
                    if "sentiment" in result and "label" in result["sentiment"]:
                        sentiment = result["sentiment"]["label"]
                        time_series[date_key]["sentiment"][sentiment] += 1
                    
                    if "entities" in result:
                        time_series[date_key]["entities"] += len(result["entities"])
                except:
                    continue
        
        # Calculate trends
        dates = sorted(time_series.keys())
        if len(dates) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        sentiment_trends = defaultdict(list)
        entity_trends = []
        
        for date in dates:
            data = time_series[date]
            total_docs = data["documents"]
            
            for sentiment, count in data["sentiment"].items():
                sentiment_trends[sentiment].append(count / total_docs if total_docs > 0 else 0)
            
            entity_trends.append(data["entities"] / total_docs if total_docs > 0 else 0)
        
        # Calculate trend directions
        trends = {}
        for sentiment, values in sentiment_trends.items():
            if values:
                trend = "increasing" if values[-1] > values[0] else "decreasing"
                trends[sentiment] = {
                    "direction": trend,
                    "start_value": values[0],
                    "end_value": values[-1],
                    "change": values[-1] - values[0]
                }
        
        return {
            "period": {"start": dates[0], "end": dates[-1]},
            "sentiment_trends": trends,
            "entity_density_trend": {
                "start": entity_trends[0] if entity_trends else 0,
                "end": entity_trends[-1] if entity_trends else 0,
                "change": entity_trends[-1] - entity_trends[0] if entity_trends else 0
            },
            "daily_averages": {
                "documents": len(self.results) / len(dates),
                "entities": sum(entity_trends) / len(entity_trends) if entity_trends else 0
            }
        }
    
    def _update_metrics(self) -> None:
        """Update aggregation metrics."""
        self.metrics.total_documents = len(self.results)
        self.metrics.successful_documents = sum(1 for r in self.results if not r.get("error"))
        self.metrics.failed_documents = self.metrics.total_documents - self.metrics.successful_documents
        
        # Sentiment metrics
        sentiment_scores = []
        sentiment_counts = Counter()
        
        for result in self.results:
            if "sentiment" in result:
                if "label" in result["sentiment"]:
                    sentiment_counts[result["sentiment"]["label"]] += 1
                if "score" in result["sentiment"]:
                    sentiment_scores.append(result["sentiment"]["score"])
        
        self.metrics.sentiment_distribution = dict(sentiment_counts)
        if sentiment_scores:
            self.metrics.sentiment_confidence_avg = statistics.mean(sentiment_scores)
            self.metrics.sentiment_confidence_std = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
        
        # Entity metrics
        all_entities = []
        entity_counts_per_doc = []
        
        for result in self.results:
            if "entities" in result:
                entities = result["entities"]
                entity_counts_per_doc.append(len(entities))
                all_entities.extend(entities)
        
        self.metrics.total_entities = len(all_entities)
        self.metrics.unique_entities = len(set(e.get("text", "") for e in all_entities))
        
        entity_types = [e.get("label", "UNKNOWN") for e in all_entities]
        self.metrics.entity_type_distribution = dict(Counter(entity_types))
        
        if entity_counts_per_doc:
            self.metrics.entities_per_document_avg = statistics.mean(entity_counts_per_doc)
        
        # Text metrics
        text_lengths = []
        word_counts = []
        
        for result in self.results:
            if "text" in result:
                text_lengths.append(len(result["text"]))
                word_counts.append(len(result["text"].split()))
        
        if text_lengths:
            self.metrics.text_length_avg = statistics.mean(text_lengths)
        if word_counts:
            self.metrics.word_count_avg = statistics.mean(word_counts)
    
    def _calculate_avg_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score."""
        scores = []
        for result in results:
            if "sentiment" in result and "score" in result["sentiment"]:
                scores.append(result["sentiment"]["score"])
        return statistics.mean(scores) if scores else 0.0
    
    def _get_common_entities(self, results: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get most common entities from results."""
        entities = []
        for result in results:
            if "entities" in result:
                entities.extend([e.get("text", "") for e in result["entities"]])
        
        entity_counts = Counter(entities).most_common(top_n)
        return [{"entity": entity, "count": count} for entity, count in entity_counts]
    
    def _calculate_text_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate text statistics."""
        lengths = []
        words = []
        
        for result in results:
            if "text" in result:
                lengths.append(len(result["text"]))
                words.append(len(result["text"].split()))
        
        return {
            "average_length": statistics.mean(lengths) if lengths else 0,
            "average_words": statistics.mean(words) if words else 0,
            "total_characters": sum(lengths),
            "total_words": sum(words)
        }
    
    def _get_sentiment_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get sentiment distribution from results."""
        sentiments = []
        for result in results:
            if "sentiment" in result and "label" in result["sentiment"]:
                sentiments.append(result["sentiment"]["label"])
        return dict(Counter(sentiments))
    
    def _get_unique_entities(self, results: List[Dict[str, Any]]) -> Set[str]:
        """Get unique entities from results."""
        entities = set()
        for result in results:
            if "entities" in result:
                entities.update(e.get("text", "") for e in result["entities"])
        return entities
    
    def _get_top_entities_by_type(self, entities: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top entities by frequency."""
        entity_counts = Counter(e["text"] for e in entities)
        return [{"text": text, "count": count} for text, count in entity_counts.most_common(top_n)]