"""spaCy-based entity extraction model."""

from typing import Dict, List, Union, Any, Optional
import spacy
from spacy.tokens import Doc
import logging
from datetime import datetime
from collections import defaultdict

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EntityExtractor(BaseModel):
    """spaCy-based named entity recognition model."""
    
    # Standard spaCy entity labels
    ENTITY_LABELS = {
        "PERSON": "Person",
        "ORG": "Organization", 
        "GPE": "Location",
        "LOC": "Location",
        "DATE": "Date",
        "TIME": "Time",
        "MONEY": "Money",
        "PERCENT": "Percentage",
        "PRODUCT": "Product",
        "EVENT": "Event",
        "FAC": "Facility",
        "LAW": "Law",
        "LANGUAGE": "Language",
        "WORK_OF_ART": "Work of Art",
        "NORP": "Nationality/Group",
        "QUANTITY": "Quantity",
        "ORDINAL": "Ordinal",
        "CARDINAL": "Cardinal"
    }
    
    def __init__(self, model_name: str = "en_core_web_sm", version: str = "1.0.0"):
        """
        Initialize entity extractor.
        
        Args:
            model_name: spaCy model name
            version: Model version
        """
        super().__init__(model_name, version)
        self._config = {
            "model_name": model_name,
            "confidence_threshold": 0.0,  # spaCy doesn't provide confidence scores by default
            "merge_entities": True
        }
        
    def load_model(self) -> None:
        """Load spaCy model."""
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            
            # Load spaCy model
            self.model = spacy.load(self.model_name)
            
            self.is_loaded = True
            self.load_timestamp = datetime.now()
            
            logger.info(f"Entity extraction model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load entity extraction model: {e}")
            raise
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract entities from input text.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Entity extraction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        texts = self.validate_input(text)
        is_single = isinstance(text, str)
        
        # Process texts
        results = []
        for doc in self.model.pipe(texts):
            result = self._extract_entities(doc, doc.text)
            results.append(result)
        
        return results[0] if is_single else results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Extract entities from a batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of entity extraction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        texts = self.validate_input(texts)
        
        # Process with spaCy pipe for efficiency
        results = []
        for doc in self.model.pipe(texts, batch_size=batch_size):
            result = self._extract_entities(doc, doc.text)
            results.append(result)
            
        return results
    
    def _extract_entities(self, doc: Doc, original_text: str) -> Dict[str, Any]:
        """
        Extract entities from a spaCy doc.
        
        Args:
            doc: spaCy Doc object
            original_text: Original input text
            
        Returns:
            Dictionary with extracted entities
        """
        entities = []
        entity_counts = defaultdict(int)
        
        for ent in doc.ents:
            entity_type = self.ENTITY_LABELS.get(ent.label_, ent.label_)
            
            entity_info = {
                "text": ent.text,
                "type": entity_type,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # spaCy doesn't provide confidence scores
            }
            
            entities.append(entity_info)
            entity_counts[entity_type] += 1
        
        # Extract additional features
        tokens = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            "text": original_text,
            "entities": entities,
            "entity_counts": dict(entity_counts),
            "total_entities": len(entities),
            "tokens": tokens,
            "pos_tags": pos_tags,
            "model_version": self.version
        }
    
    def extract_entity_types(self, texts: List[str], 
                           entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Extract specific entity types from multiple texts.
        
        Args:
            texts: List of texts
            entity_types: List of entity types to extract (None for all)
            
        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        results = self.predict_batch(texts)
        
        # Collect entities by type
        entities_by_type = defaultdict(set)
        
        for result in results:
            for entity in result["entities"]:
                if entity_types is None or entity["type"] in entity_types:
                    entities_by_type[entity["type"]].add(entity["text"])
        
        # Convert sets to sorted lists
        return {
            entity_type: sorted(list(entities))
            for entity_type, entities in entities_by_type.items()
        }
    
    def find_entity_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Find potential relationships between entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential entity relationships
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        doc = self.model(text)
        relationships = []
        
        # Simple relationship extraction based on sentence co-occurrence
        for sent in doc.sents:
            sent_entities = [ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end]
            
            # Find pairs of entities in the same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # Look for verbs between entities
                    start_token = min(ent1.end, ent2.end)
                    end_token = max(ent1.start, ent2.start)
                    
                    if start_token < end_token:
                        between_tokens = doc[start_token:end_token]
                        verbs = [token.text for token in between_tokens if token.pos_ == "VERB"]
                        
                        if verbs:
                            relationships.append({
                                "entity1": {"text": ent1.text, "type": ent1.label_},
                                "entity2": {"text": ent2.text, "type": ent2.label_},
                                "relation": " ".join(verbs),
                                "sentence": sent.text.strip()
                            })
        
        return relationships
    
    def get_entity_context(self, text: str, window_size: int = 50) -> List[Dict[str, Any]]:
        """
        Get context around each entity.
        
        Args:
            text: Input text
            window_size: Number of characters around entity to include
            
        Returns:
            List of entities with their context
        """
        result = self.predict(text)
        entities_with_context = []
        
        for entity in result["entities"]:
            start = max(0, entity["start"] - window_size)
            end = min(len(text), entity["end"] + window_size)
            
            context = text[start:end]
            entity_start_in_context = entity["start"] - start
            entity_end_in_context = entity["end"] - start
            
            entities_with_context.append({
                "entity": entity,
                "context": context,
                "entity_position": (entity_start_in_context, entity_end_in_context)
            })
        
        return entities_with_context