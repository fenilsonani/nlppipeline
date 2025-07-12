"""Generate comprehensive analysis reports from NLP pipeline results."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter, defaultdict
import statistics


class ReportGenerator:
    """Generate various analysis reports from processed documents."""
    
    def __init__(self):
        """Initialize report generator."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: List of document processing results
            
        Returns:
            Dictionary containing summary statistics and insights
        """
        if not results:
            return {"error": "No results to analyze"}
        
        report = {
            "metadata": {
                "total_documents": len(results),
                "report_generated": datetime.now().isoformat(),
                "processing_summary": self._get_processing_summary(results)
            },
            "sentiment_analysis": self._analyze_sentiment_distribution(results),
            "entity_analysis": self._analyze_entities(results),
            "text_statistics": self._calculate_text_statistics(results),
            "quality_metrics": self._calculate_quality_metrics(results)
        }
        
        return report
    
    def generate_sentiment_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate detailed sentiment analysis report.
        
        Args:
            results: List of document processing results
            
        Returns:
            Dictionary containing sentiment analysis details
        """
        sentiment_data = []
        
        for result in results:
            if "sentiment" in result:
                sentiment_data.append({
                    "document_id": result.get("document_id", "unknown"),
                    "sentiment": result["sentiment"].get("label", "neutral"),
                    "confidence": result["sentiment"].get("score", 0.0),
                    "text_preview": result.get("text", "")[:100] + "..."
                })
        
        # Calculate distribution
        sentiments = [d["sentiment"] for d in sentiment_data]
        distribution = dict(Counter(sentiments))
        
        # Calculate average confidence by sentiment
        confidence_by_sentiment = defaultdict(list)
        for data in sentiment_data:
            confidence_by_sentiment[data["sentiment"]].append(data["confidence"])
        
        avg_confidence = {
            sentiment: statistics.mean(scores) if scores else 0
            for sentiment, scores in confidence_by_sentiment.items()
        }
        
        return {
            "total_analyzed": len(sentiment_data),
            "distribution": distribution,
            "average_confidence": avg_confidence,
            "detailed_results": sentiment_data,
            "insights": self._generate_sentiment_insights(distribution, avg_confidence)
        }
    
    def generate_entity_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate detailed entity analysis report.
        
        Args:
            results: List of document processing results
            
        Returns:
            Dictionary containing entity analysis details
        """
        all_entities = []
        entity_by_type = defaultdict(list)
        
        for result in results:
            if "entities" in result:
                for entity in result["entities"]:
                    entity_info = {
                        "text": entity.get("text", ""),
                        "type": entity.get("label", "UNKNOWN"),
                        "confidence": entity.get("score", 0.0),
                        "document_id": result.get("document_id", "unknown")
                    }
                    all_entities.append(entity_info)
                    entity_by_type[entity_info["type"]].append(entity_info["text"])
        
        # Calculate frequency for each entity type
        entity_frequencies = {}
        for entity_type, entities in entity_by_type.items():
            entity_frequencies[entity_type] = dict(Counter(entities).most_common(20))
        
        return {
            "total_entities": len(all_entities),
            "unique_entities": len(set(e["text"] for e in all_entities)),
            "entity_types": list(entity_by_type.keys()),
            "frequencies_by_type": entity_frequencies,
            "top_entities": self._get_top_entities(all_entities, n=50),
            "entity_coverage": self._calculate_entity_coverage(results),
            "insights": self._generate_entity_insights(entity_frequencies)
        }
    
    def generate_document_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate report for a single document.
        
        Args:
            result: Single document processing result
            
        Returns:
            Dictionary containing document-specific analysis
        """
        report = {
            "document_id": result.get("document_id", "unknown"),
            "metadata": {
                "processed_at": result.get("processed_at", datetime.now().isoformat()),
                "text_length": len(result.get("text", "")),
                "word_count": len(result.get("text", "").split())
            }
        }
        
        if "sentiment" in result:
            report["sentiment"] = {
                "label": result["sentiment"].get("label"),
                "confidence": result["sentiment"].get("score"),
                "sentiment_segments": self._analyze_sentiment_segments(result.get("text", ""))
            }
        
        if "entities" in result:
            report["entities"] = {
                "total_found": len(result["entities"]),
                "by_type": self._group_entities_by_type(result["entities"]),
                "entity_density": len(result["entities"]) / max(1, report["metadata"]["word_count"])
            }
        
        if "classification" in result:
            report["classification"] = result["classification"]
        
        return report
    
    def _get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get processing summary statistics."""
        successful = sum(1 for r in results if not r.get("error"))
        failed = len(results) - successful
        
        return {
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0
        }
    
    def _analyze_sentiment_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment distribution across documents."""
        sentiments = []
        for result in results:
            if "sentiment" in result and "label" in result["sentiment"]:
                sentiments.append(result["sentiment"]["label"])
        
        distribution = dict(Counter(sentiments))
        total = len(sentiments)
        
        return {
            "distribution": distribution,
            "percentages": {k: v/total*100 for k, v in distribution.items()} if total > 0 else {},
            "dominant_sentiment": max(distribution.items(), key=lambda x: x[1])[0] if distribution else None
        }
    
    def _analyze_entities(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entities across documents."""
        all_entities = []
        for result in results:
            if "entities" in result:
                all_entities.extend(result["entities"])
        
        entity_types = [e.get("label", "UNKNOWN") for e in all_entities]
        type_distribution = dict(Counter(entity_types))
        
        return {
            "total_entities": len(all_entities),
            "unique_entities": len(set(e.get("text", "") for e in all_entities)),
            "type_distribution": type_distribution,
            "entities_per_document": len(all_entities) / len(results) if results else 0
        }
    
    def _calculate_text_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate text statistics across documents."""
        text_lengths = []
        word_counts = []
        
        for result in results:
            if "text" in result:
                text_lengths.append(len(result["text"]))
                word_counts.append(len(result["text"].split()))
        
        return {
            "average_length": statistics.mean(text_lengths) if text_lengths else 0,
            "average_words": statistics.mean(word_counts) if word_counts else 0,
            "total_characters": sum(text_lengths),
            "total_words": sum(word_counts)
        }
    
    def _calculate_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the processing."""
        sentiment_confidences = []
        entity_confidences = []
        
        for result in results:
            if "sentiment" in result and "score" in result["sentiment"]:
                sentiment_confidences.append(result["sentiment"]["score"])
            
            if "entities" in result:
                for entity in result["entities"]:
                    if "score" in entity:
                        entity_confidences.append(entity["score"])
        
        return {
            "average_sentiment_confidence": statistics.mean(sentiment_confidences) if sentiment_confidences else 0,
            "average_entity_confidence": statistics.mean(entity_confidences) if entity_confidences else 0,
            "high_confidence_sentiments": sum(1 for c in sentiment_confidences if c > 0.8) / len(sentiment_confidences) if sentiment_confidences else 0,
            "high_confidence_entities": sum(1 for c in entity_confidences if c > 0.8) / len(entity_confidences) if entity_confidences else 0
        }
    
    def _get_top_entities(self, entities: List[Dict[str, Any]], n: int = 50) -> List[Dict[str, Any]]:
        """Get top N most frequent entities."""
        entity_counts = Counter(e["text"] for e in entities)
        top_entities = []
        
        for entity, count in entity_counts.most_common(n):
            # Find the most common type for this entity
            types = [e["type"] for e in entities if e["text"] == entity]
            most_common_type = Counter(types).most_common(1)[0][0]
            
            top_entities.append({
                "text": entity,
                "type": most_common_type,
                "frequency": count
            })
        
        return top_entities
    
    def _calculate_entity_coverage(self, results: List[Dict[str, Any]]) -> float:
        """Calculate what percentage of documents contain entities."""
        docs_with_entities = sum(1 for r in results if r.get("entities"))
        return docs_with_entities / len(results) if results else 0
    
    def _generate_sentiment_insights(self, distribution: Dict[str, int], 
                                   avg_confidence: Dict[str, float]) -> List[str]:
        """Generate insights from sentiment analysis."""
        insights = []
        
        if distribution:
            dominant = max(distribution.items(), key=lambda x: x[1])
            insights.append(f"Dominant sentiment is '{dominant[0]}' ({dominant[1]} documents)")
            
            if "positive" in distribution and "negative" in distribution:
                ratio = distribution["positive"] / (distribution["negative"] + 1)
                insights.append(f"Positive to negative ratio: {ratio:.2f}")
        
        if avg_confidence:
            high_conf = [s for s, c in avg_confidence.items() if c > 0.8]
            if high_conf:
                insights.append(f"High confidence sentiments: {', '.join(high_conf)}")
        
        return insights
    
    def _generate_entity_insights(self, entity_frequencies: Dict[str, Dict[str, int]]) -> List[str]:
        """Generate insights from entity analysis."""
        insights = []
        
        if entity_frequencies:
            most_common_type = max(entity_frequencies.items(), 
                                 key=lambda x: sum(x[1].values()))
            insights.append(f"Most prevalent entity type: {most_common_type[0]}")
            
            for entity_type, frequencies in entity_frequencies.items():
                if frequencies:
                    top_entity = max(frequencies.items(), key=lambda x: x[1])
                    insights.append(f"Top {entity_type}: '{top_entity[0]}' ({top_entity[1]} occurrences)")
        
        return insights
    
    def _analyze_sentiment_segments(self, text: str, segment_size: int = 100) -> List[Dict[str, Any]]:
        """Analyze sentiment in text segments (placeholder for actual implementation)."""
        # This would require re-analyzing segments of text
        # For now, return empty list as this would need the sentiment analyzer
        return []
    
    def _group_entities_by_type(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group entities by their type."""
        grouped = defaultdict(list)
        for entity in entities:
            grouped[entity.get("label", "UNKNOWN")].append(entity.get("text", ""))
        return dict(grouped)