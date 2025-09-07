"""
Advanced drift detection for llmentary

This module provides sophisticated drift detection capabilities including:
- Semantic similarity analysis using sentence transformers
- Configurable drift thresholds
- Severity classification
- Trend analysis and pattern recognition
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime, timedelta
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from .llmentary import LLMCall, logger
except ImportError:
    from llmentary import LLMCall, logger

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class DriftType(Enum):
    """Types of drift detection"""
    EXACT = "exact"          # Hash-based exact matching
    SEMANTIC = "semantic"    # Semantic similarity based
    HYBRID = "hybrid"        # Combination of both

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""
    # Thresholds (0.0 = no similarity, 1.0 = identical)
    exact_threshold: float = 1.0          # Must be exactly the same
    semantic_threshold: float = 0.85      # Semantic similarity threshold
    hybrid_threshold: float = 0.80        # Combined threshold for hybrid mode
    
    # Severity thresholds
    low_threshold: float = 0.7
    medium_threshold: float = 0.5
    high_threshold: float = 0.3
    
    # Detection mode
    detection_mode: DriftType = DriftType.HYBRID
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast and efficient
    cache_embeddings: bool = True
    max_embedding_cache: int = 10000

@dataclass 
class DriftDetectionResult:
    """Result of drift detection analysis"""
    input_hash: str
    has_drift: bool
    drift_type: DriftType
    severity: DriftSeverity
    similarity_score: float
    confidence: float
    previous_outputs: List[str]
    current_output: str
    metadata: Dict[str, Any]
    
class AdvancedDriftDetector:
    """Advanced drift detection with semantic analysis"""
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        self.embedding_model = None
        self.embedding_cache = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, semantic analysis disabled")
            return
            
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Initialized embedding model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching"""
        if not self.embedding_model:
            return None
            
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if self.config.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            embedding = self.embedding_model.encode([text])[0]
            
            # Cache the embedding
            if self.config.cache_embeddings:
                if len(self.embedding_cache) >= self.config.max_embedding_cache:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
                
                self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
    
    def calculate_exact_similarity(self, text1: str, text2: str) -> float:
        """Calculate exact similarity (1.0 if identical, 0.0 otherwise)"""
        return 1.0 if text1.strip() == text2.strip() else 0.0
    
    def calculate_hybrid_similarity(self, text1: str, text2: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate hybrid similarity combining exact and semantic"""
        exact_sim = self.calculate_exact_similarity(text1, text2)
        semantic_sim = self.calculate_semantic_similarity(text1, text2)
        
        # If exact match, return 1.0
        if exact_sim == 1.0:
            return 1.0, {"exact": 1.0, "semantic": 1.0, "method": "exact"}
        
        # Otherwise use semantic similarity with slight boost for near-exact matches
        if semantic_sim > 0.95:
            # Boost very high semantic similarity slightly
            hybrid_score = min(1.0, semantic_sim + 0.05)
        else:
            hybrid_score = semantic_sim
            
        return hybrid_score, {
            "exact": exact_sim,
            "semantic": semantic_sim,
            "hybrid": hybrid_score,
            "method": "hybrid"
        }
    
    def classify_drift_severity(self, similarity_score: float) -> DriftSeverity:
        """Classify drift severity based on similarity score"""
        if similarity_score >= self.config.low_threshold:
            return DriftSeverity.LOW
        elif similarity_score >= self.config.medium_threshold:
            return DriftSeverity.MEDIUM
        elif similarity_score >= self.config.high_threshold:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def detect_drift(self, current_call: LLMCall, previous_calls: List[LLMCall]) -> DriftDetectionResult:
        """Detect drift for a given call against previous calls"""
        if not previous_calls:
            return DriftDetectionResult(
                input_hash=current_call.input_hash,
                has_drift=False,
                drift_type=self.config.detection_mode,
                severity=DriftSeverity.LOW,
                similarity_score=1.0,
                confidence=1.0,
                previous_outputs=[],
                current_output=current_call.output_text or "",
                metadata={"reason": "no_previous_calls"}
            )
        
        current_output = current_call.output_text or ""
        previous_outputs = [call.output_text or "" for call in previous_calls]
        
        # Calculate similarities against all previous outputs
        similarities = []
        similarity_details = []
        
        for prev_output in previous_outputs:
            if self.config.detection_mode == DriftType.EXACT:
                sim = self.calculate_exact_similarity(current_output, prev_output)
                details = {"exact": sim, "method": "exact"}
            elif self.config.detection_mode == DriftType.SEMANTIC:
                sim = self.calculate_semantic_similarity(current_output, prev_output)
                details = {"semantic": sim, "method": "semantic"}
            else:  # HYBRID
                sim, details = self.calculate_hybrid_similarity(current_output, prev_output)
            
            similarities.append(sim)
            similarity_details.append(details)
        
        # Use the highest similarity (most similar to any previous output)
        max_similarity = max(similarities) if similarities else 0.0
        max_index = similarities.index(max_similarity) if similarities else 0
        
        # Determine if there's drift based on the appropriate threshold
        if self.config.detection_mode == DriftType.EXACT:
            threshold = self.config.exact_threshold
        elif self.config.detection_mode == DriftType.SEMANTIC:
            threshold = self.config.semantic_threshold
        else:  # HYBRID
            threshold = self.config.hybrid_threshold
        
        has_drift = max_similarity < threshold
        severity = self.classify_drift_severity(max_similarity)
        
        # Calculate confidence based on consistency of similarities
        if len(similarities) > 1:
            std_dev = np.std(similarities)
            confidence = max(0.0, min(1.0, 1.0 - std_dev))
        else:
            confidence = 1.0
        
        return DriftDetectionResult(
            input_hash=current_call.input_hash,
            has_drift=has_drift,
            drift_type=self.config.detection_mode,
            severity=severity,
            similarity_score=max_similarity,
            confidence=confidence,
            previous_outputs=previous_outputs,
            current_output=current_output,
            metadata={
                "threshold_used": threshold,
                "all_similarities": similarities,
                "similarity_details": similarity_details[max_index] if similarity_details else {},
                "num_previous_calls": len(previous_calls),
                "model": current_call.model,
                "provider": current_call.provider
            }
        )
    
    def analyze_drift_patterns(self, calls: List[LLMCall], time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze drift patterns across a set of calls"""
        if not calls:
            return {"error": "No calls provided"}
        
        # Group calls by input hash
        input_groups = {}
        for call in calls:
            if call.input_hash not in input_groups:
                input_groups[call.input_hash] = []
            input_groups[call.input_hash].append(call)
        
        # Sort calls in each group by timestamp
        for input_hash in input_groups:
            input_groups[input_hash].sort(key=lambda x: x.timestamp)
        
        # Analyze each group for drift
        drift_results = []
        severity_counts = {sev.value: 0 for sev in DriftSeverity}
        
        for input_hash, group_calls in input_groups.items():
            if len(group_calls) < 2:
                continue
                
            # Analyze each call against previous calls in the group
            for i in range(1, len(group_calls)):
                current_call = group_calls[i]
                previous_calls = group_calls[:i]
                
                # Filter to time window if specified
                if time_window_hours:
                    cutoff_time = current_call.timestamp - timedelta(hours=time_window_hours)
                    previous_calls = [c for c in previous_calls if c.timestamp >= cutoff_time]
                
                if not previous_calls:
                    continue
                
                result = self.detect_drift(current_call, previous_calls)
                drift_results.append(result)
                
                if result.has_drift:
                    severity_counts[result.severity.value] += 1
        
        # Calculate statistics
        total_comparisons = len(drift_results)
        drift_cases = sum(1 for r in drift_results if r.has_drift)
        
        if total_comparisons == 0:
            drift_rate = 0.0
        else:
            drift_rate = drift_cases / total_comparisons
        
        # Average similarity score
        avg_similarity = np.mean([r.similarity_score for r in drift_results]) if drift_results else 0.0
        
        # Model and provider breakdown
        model_drift_rates = {}
        provider_drift_rates = {}
        
        for result in drift_results:
            model = result.metadata.get('model', 'unknown')
            provider = result.metadata.get('provider', 'unknown')
            
            if model not in model_drift_rates:
                model_drift_rates[model] = {"total": 0, "drift": 0}
            if provider not in provider_drift_rates:
                provider_drift_rates[provider] = {"total": 0, "drift": 0}
            
            model_drift_rates[model]["total"] += 1
            provider_drift_rates[provider]["total"] += 1
            
            if result.has_drift:
                model_drift_rates[model]["drift"] += 1
                provider_drift_rates[provider]["drift"] += 1
        
        # Calculate rates
        for model_stats in model_drift_rates.values():
            model_stats["rate"] = model_stats["drift"] / model_stats["total"] if model_stats["total"] > 0 else 0.0
            
        for provider_stats in provider_drift_rates.values():
            provider_stats["rate"] = provider_stats["drift"] / provider_stats["total"] if provider_stats["total"] > 0 else 0.0
        
        return {
            "summary": {
                "total_comparisons": total_comparisons,
                "drift_cases": drift_cases,
                "drift_rate": drift_rate,
                "average_similarity": float(avg_similarity),
                "detection_mode": self.config.detection_mode.value,
                "time_window_hours": time_window_hours
            },
            "severity_breakdown": severity_counts,
            "model_drift_rates": model_drift_rates,
            "provider_drift_rates": provider_drift_rates,
            "drift_results": [
                {
                    "input_hash": r.input_hash[:8],
                    "has_drift": r.has_drift,
                    "severity": r.severity.value,
                    "similarity_score": r.similarity_score,
                    "confidence": r.confidence,
                    "model": r.metadata.get('model'),
                    "provider": r.metadata.get('provider')
                }
                for r in drift_results if r.has_drift
            ]
        }

# Global detector instance
_detector = None

def get_detector(config: DriftDetectionConfig = None) -> AdvancedDriftDetector:
    """Get the global drift detector instance"""
    global _detector
    if _detector is None or config is not None:
        _detector = AdvancedDriftDetector(config)
    return _detector