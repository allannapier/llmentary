"""
llmentary - Privacy-first LLM monitoring and regression testing

A comprehensive toolkit for LLM applications that provides:
1. Privacy-first monitoring with hash-based drift detection
2. Training/testing workflow for regression testing
3. Advanced semantic similarity analysis
4. Minimal integration overhead
"""

import hashlib
import json
import sqlite3
import numpy as np
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

# Optional dependencies with graceful fallbacks
try:
    import structlog

    logger = structlog.get_logger()
except ImportError:

    class SimpleLogger:
        def info(self, msg, **kwargs):
            print(f"INFO: {msg}")

        def warning(self, msg, **kwargs):
            print(f"WARNING: {msg}")

        def error(self, msg, **kwargs):
            print(f"ERROR: {msg}")

    logger = SimpleLogger()

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


@dataclass
class LLMCall:
    """Represents a single LLM call"""

    input_hash: str
    output_hash: str
    input_text: Optional[str]
    output_text: Optional[str]
    model: str
    provider: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class TrainingExample:
    """A saved question-answer pair for regression testing"""

    id: Optional[int]
    question: str  # Stored in plaintext
    answer_hash: str  # Hashed for privacy
    model: str
    provider: str
    category: Optional[str]
    tags: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "question": self.question,
            "answer_hash": self.answer_hash,
            "model": self.model,
            "provider": self.provider,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """Result of testing a training example against current response"""

    example_id: int
    question: str
    category: Optional[str]
    expected_hash: str
    actual_hash: str
    matches: bool
    similarity_score: Optional[float]
    model: str
    provider: str
    tested_at: datetime
    metadata: Dict[str, Any]


class DriftSeverity(Enum):
    """Drift severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift detection"""

    EXACT = "exact"  # Hash-based exact matching
    SEMANTIC = "semantic"  # Semantic similarity based
    HYBRID = "hybrid"  # Combination of both


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""

    # Thresholds (0.0 = no similarity, 1.0 = identical)
    exact_threshold: float = 1.0  # Must be exactly the same
    semantic_threshold: float = 0.85  # Semantic similarity threshold
    hybrid_threshold: float = 0.80  # Combined threshold for hybrid mode

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


# =============================================================================
# STORAGE LAYER
# =============================================================================


class SQLiteStorage:
    """SQLite storage backend for LLM calls"""

    def __init__(self, db_path: str = "llmentary.db"):
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_input_hash ON llm_calls(input_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_output_hash ON llm_calls(output_hash)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON llm_calls(model)")

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def store_call(self, call: LLMCall):
        """Store an LLM call"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_calls 
                (input_hash, output_hash, input_text, output_text, model, provider, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    call.input_hash,
                    call.output_hash,
                    call.input_text,
                    call.output_text,
                    call.model,
                    call.provider,
                    call.timestamp.isoformat(),
                    json.dumps(call.metadata),
                ),
            )
            conn.commit()

    def get_calls_by_input_hash(self, input_hash: str) -> List[LLMCall]:
        """Get all calls for a given input hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM llm_calls WHERE input_hash = ? ORDER BY timestamp",
                (input_hash,),
            )

            calls = []
            for row in cursor.fetchall():
                calls.append(
                    LLMCall(
                        input_hash=row[1],
                        output_hash=row[2],
                        input_text=row[3],
                        output_text=row[4],
                        model=row[5],
                        provider=row[6],
                        timestamp=datetime.fromisoformat(row[7]),
                        metadata=json.loads(row[8] or "{}"),
                    )
                )

            return calls


class TrainingTestStorage:
    """SQLite storage for training examples and test results"""

    def __init__(self, db_path: str = "llmentary_training.db"):
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the training database"""
        with sqlite3.connect(self.db_path) as conn:
            # Training examples table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    category TEXT,
                    tags TEXT,  -- JSON array
                    created_at TEXT NOT NULL,
                    metadata TEXT  -- JSON object
                )
            """)

            # Test results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id INTEGER NOT NULL,
                    expected_hash TEXT NOT NULL,
                    actual_hash TEXT NOT NULL,
                    matches BOOLEAN NOT NULL,
                    similarity_score REAL,
                    tested_at TEXT NOT NULL,
                    metadata TEXT,  -- JSON object
                    FOREIGN KEY (example_id) REFERENCES training_examples (id)
                )
            """)

            # Indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_category ON training_examples(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model ON training_examples(model)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider ON training_examples(provider)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_example ON test_results(example_id)"
            )

            conn.commit()
            logger.info(f"Training database initialized at {self.db_path}")

    def save_example(self, example: TrainingExample) -> int:
        """Save a training example and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_examples 
                (question, answer_hash, model, provider, category, tags, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    example.question,
                    example.answer_hash,
                    example.model,
                    example.provider,
                    example.category,
                    json.dumps(example.tags),
                    example.created_at.isoformat(),
                    json.dumps(example.metadata),
                ),
            )

            example_id = cursor.lastrowid
            conn.commit()
            logger.info(
                f"Saved training example {example_id}",
                question=example.question[:50] + "..."
                if len(example.question) > 50
                else example.question,
            )
            return example_id

    def get_examples(
        self,
        category: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[TrainingExample]:
        """Retrieve training examples with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM training_examples WHERE 1=1"
            params = []

            if category:
                query += " AND category = ?"
                params.append(category)
            if model:
                query += " AND model = ?"
                params.append(model)
            if provider:
                query += " AND provider = ?"
                params.append(provider)

            query += " ORDER BY created_at DESC"

            cursor = conn.cursor()
            cursor.execute(query, params)

            examples = []
            for row in cursor.fetchall():
                example_tags = json.loads(row[6] or "[]")

                # Filter by tags if specified
                if tags and not any(tag in example_tags for tag in tags):
                    continue

                examples.append(
                    TrainingExample(
                        id=row[0],
                        question=row[1],
                        answer_hash=row[2],
                        model=row[3],
                        provider=row[4],
                        category=row[5],
                        tags=example_tags,
                        created_at=datetime.fromisoformat(row[7]),
                        metadata=json.loads(row[8] or "{}"),
                    )
                )

            return examples

    def save_test_result(self, result: TestResult) -> int:
        """Save a test result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO test_results 
                (example_id, expected_hash, actual_hash, matches, similarity_score, tested_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.example_id,
                    result.expected_hash,
                    result.actual_hash,
                    result.matches,
                    result.similarity_score,
                    result.tested_at.isoformat(),
                    json.dumps(result.metadata),
                ),
            )

            result_id = cursor.lastrowid
            conn.commit()
            return result_id


# =============================================================================
# ADVANCED DRIFT DETECTION
# =============================================================================


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
            logger.warning(
                "sentence-transformers not available, semantic analysis disabled"
            )
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

        if self.config.cache_embeddings and text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            embedding = self.embedding_model.encode([text])[0]

            if self.config.cache_embeddings:
                # Simple cache eviction when full
                if len(self.embedding_cache) >= self.config.max_embedding_cache:
                    # Remove oldest 25% of entries (simple FIFO)
                    items_to_remove = list(self.embedding_cache.keys())[
                        : len(self.embedding_cache) // 4
                    ]
                    for key in items_to_remove:
                        del self.embedding_cache[key]

                self.embedding_cache[text] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _determine_severity(self, similarity_score: float) -> DriftSeverity:
        """Determine drift severity based on similarity score"""
        if similarity_score >= self.config.low_threshold:
            return DriftSeverity.LOW
        elif similarity_score >= self.config.medium_threshold:
            return DriftSeverity.MEDIUM
        elif similarity_score >= self.config.high_threshold:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def detect_drift(
        self, current_output: str, previous_outputs: List[str], input_hash: str = ""
    ) -> DriftDetectionResult:
        """Detect drift using configured method"""

        if not previous_outputs:
            return DriftDetectionResult(
                input_hash=input_hash,
                has_drift=False,
                drift_type=self.config.detection_mode,
                severity=DriftSeverity.LOW,
                similarity_score=1.0,
                confidence=1.0,
                previous_outputs=[],
                current_output=current_output,
                metadata={"reason": "no_previous_outputs"},
            )

        # Calculate similarity scores against all previous outputs
        similarities = []

        for prev_output in previous_outputs:
            if self.config.detection_mode == DriftType.EXACT:
                # Exact hash comparison
                similarity = (
                    1.0 if current_output.strip() == prev_output.strip() else 0.0
                )
            elif self.config.detection_mode == DriftType.SEMANTIC:
                # Semantic similarity
                similarity = self.semantic_similarity(current_output, prev_output)
            else:  # HYBRID
                # Combination of exact and semantic
                exact_match = (
                    1.0 if current_output.strip() == prev_output.strip() else 0.0
                )
                semantic_sim = self.semantic_similarity(current_output, prev_output)
                similarity = max(
                    exact_match, semantic_sim * 0.9
                )  # Slight penalty for non-exact

            similarities.append(similarity)

        # Use best similarity score (most similar to any previous output)
        best_similarity = max(similarities) if similarities else 0.0

        # Determine if drift occurred based on threshold
        threshold_map = {
            DriftType.EXACT: self.config.exact_threshold,
            DriftType.SEMANTIC: self.config.semantic_threshold,
            DriftType.HYBRID: self.config.hybrid_threshold,
        }

        threshold = threshold_map[self.config.detection_mode]
        has_drift = best_similarity < threshold

        # Calculate confidence (how certain we are about the result)
        confidence = (
            abs(best_similarity - threshold) / threshold if threshold > 0 else 1.0
        )
        confidence = min(1.0, max(0.0, confidence))

        return DriftDetectionResult(
            input_hash=input_hash,
            has_drift=has_drift,
            drift_type=self.config.detection_mode,
            severity=self._determine_severity(best_similarity),
            similarity_score=best_similarity,
            confidence=confidence,
            previous_outputs=previous_outputs,
            current_output=current_output,
            metadata={
                "all_similarities": similarities,
                "threshold_used": threshold,
                "detection_mode": self.config.detection_mode.value,
            },
        )


# =============================================================================
# MONITORING SYSTEM
# =============================================================================


class Monitor:
    """Main monitoring class"""

    def __init__(self):
        self.config = {
            "store_raw_text": False,
            "drift_threshold": 0.85,
            "storage_backend": "sqlite",
            "db_path": "llmentary.db",
            "advanced_drift_detection": False,
            "training_db_path": "llmentary_training.db",
        }
        self.storage = None
        self.drift_detector = None
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage backend"""
        if self.config["storage_backend"] == "sqlite":
            self.storage = SQLiteStorage(self.config["db_path"])
        else:
            raise ValueError(
                f"Unknown storage backend: {self.config['storage_backend']}"
            )

    def _initialize_drift_detector(self):
        """Initialize advanced drift detector"""
        if not self.drift_detector:
            config = DriftDetectionConfig()
            self.drift_detector = AdvancedDriftDetector(config)

    def configure(self, **kwargs):
        """Configure the monitor"""
        self.config.update(kwargs)

        # Re-initialize storage if db_path changed
        if "db_path" in kwargs and self.storage:
            self._initialize_storage()

        # Initialize drift detector if advanced detection enabled
        if self.config.get("advanced_drift_detection"):
            self._initialize_drift_detector()

        logger.info("Monitor configured")

    def _hash_text(self, text: str) -> str:
        """Create a hash of the input/output text"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def record_call(
        self,
        input_text: str,
        output_text: str,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DriftDetectionResult]:
        """Record an LLM call and check for drift"""

        input_hash = self._hash_text(input_text)
        output_hash = self._hash_text(output_text)

        # Create LLM call object
        call = LLMCall(
            input_hash=input_hash,
            output_hash=output_hash,
            input_text=input_text if self.config["store_raw_text"] else None,
            output_text=output_text if self.config["store_raw_text"] else None,
            model=model,
            provider=provider,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Store the call
        self.storage.store_call(call)

        # Check for drift
        previous_calls = self.storage.get_calls_by_input_hash(input_hash)
        if len(previous_calls) > 1:  # We have previous calls for this input
            previous_outputs = [
                c.output_text or "" for c in previous_calls[:-1]
            ]  # Exclude current

            if self.config.get("advanced_drift_detection") and self.drift_detector:
                # Use advanced drift detection
                drift_result = self.drift_detector.detect_drift(
                    current_output=output_text,
                    previous_outputs=previous_outputs,
                    input_hash=input_hash,
                )

                if drift_result.has_drift:
                    logger.warning(
                        "Advanced drift detected",
                        input_hash=input_hash[:16],
                        similarity_score=drift_result.similarity_score,
                        severity=drift_result.severity.value,
                        confidence=drift_result.confidence,
                    )

                return drift_result
            else:
                # Simple hash-based drift detection
                for prev_call in previous_calls[:-1]:
                    if prev_call.output_hash != output_hash:
                        logger.warning(
                            f"Drift detected for input hash {input_hash[:16]}"
                        )
                        break

        return None


# =============================================================================
# TRAINING/TESTING FRAMEWORK
# =============================================================================


class TrainingTester:
    """Main class for training/testing workflow"""

    def __init__(self, db_path: str = "llmentary_training.db"):
        self.storage = TrainingTestStorage(db_path)
        self.training_mode = False
        self.interactive = True

    def _hash_answer(self, answer: str, salt: str = "llmentary") -> str:
        """Create a salted hash of an answer"""
        salted = f"{salt}:{answer.strip()}"
        return hashlib.sha256(salted.encode("utf-8")).hexdigest()

    def _prompt_user_save(
        self, question: str, answer: str, model: str, provider: str
    ) -> bool:
        """Prompt user to save this Q&A pair"""
        if not self.interactive:
            return True

        if CLICK_AVAILABLE:
            click.echo(f"\n📝 Training Question: {question}")
            click.echo(
                f"🤖 Response: {answer[:200]}{'...' if len(answer) > 200 else ''}"
            )
            click.echo(f"📊 Model: {model} | Provider: {provider}")
            return click.confirm("💾 Save this Q&A pair for regression testing?")
        else:
            # Fallback without click
            print(f"\n📝 Training Question: {question}")
            print(f"🤖 Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"📊 Model: {model} | Provider: {provider}")
            response = input("💾 Save this Q&A pair for regression testing? (y/n): ")
            return response.lower().strip() in ["y", "yes"]

    def capture_interaction(
        self,
        question: str,
        answer: str,
        model: str = "unknown",
        provider: str = "unknown",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_save: bool = False,
    ) -> Optional[int]:
        """Capture a Q&A interaction, optionally prompting user to save"""

        if self.training_mode:
            should_save = force_save or self._prompt_user_save(
                question, answer, model, provider
            )

            if should_save:
                example = TrainingExample(
                    id=None,
                    question=question,
                    answer_hash=self._hash_answer(answer),
                    model=model,
                    provider=provider,
                    category=category,
                    tags=tags or [],
                    created_at=datetime.now(),
                    metadata=metadata or {},
                )

                return self.storage.save_example(example)

        return None

    def test_example(self, example: TrainingExample, current_answer: str) -> TestResult:
        """Test a single example against current answer"""
        actual_hash = self._hash_answer(current_answer)
        matches = actual_hash == example.answer_hash

        # TODO: Add semantic similarity testing here if needed
        similarity_score = 1.0 if matches else 0.0

        result = TestResult(
            example_id=example.id,
            question=example.question,
            category=example.category,
            expected_hash=example.answer_hash,
            actual_hash=actual_hash,
            matches=matches,
            similarity_score=similarity_score,
            model=example.model,
            provider=example.provider,
            tested_at=datetime.now(),
            metadata={},
        )

        self.storage.save_test_result(result)
        return result

    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about training data"""
        examples = self.storage.get_examples()

        if not examples:
            return {"total": 0}

        categories = {}
        models = {}
        providers = {}

        for example in examples:
            # Count by category
            cat = example.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1

            # Count by model
            models[example.model] = models.get(example.model, 0) + 1

            # Count by provider
            providers[example.provider] = providers.get(example.provider, 0) + 1

        return {
            "total": len(examples),
            "categories": categories,
            "models": models,
            "providers": providers,
            "oldest": min(e.created_at for e in examples),
            "newest": max(e.created_at for e in examples),
        }


# =============================================================================
# AUTO-INSTRUMENTATION
# =============================================================================


class AutoInstrument:
    """Automatic instrumentation for LLM providers"""

    @staticmethod
    def auto_patch_all():
        """Automatically patch all supported LLM providers"""
        AutoInstrument._patch_openai()
        AutoInstrument._patch_anthropic()
        AutoInstrument._patch_gemini()

    @staticmethod
    def _patch_openai():
        """Patch OpenAI client"""
        try:
            import openai

            original_create = openai.OpenAI().chat.completions.create

            def patched_create(*args, **kwargs):
                response = original_create(*args, **kwargs)

                # Extract input/output for monitoring
                messages = kwargs.get("messages", [])
                if messages:
                    input_text = messages[-1].get("content", "")
                    output_text = (
                        response.choices[0].message.content if response.choices else ""
                    )
                    model = kwargs.get("model", "unknown")

                    monitor.record_call(input_text, output_text, model, "openai")

                return response

            openai.OpenAI().chat.completions.create = patched_create
            logger.info("OpenAI patched successfully")

        except ImportError:
            logger.warning("OpenAI not available for patching")
        except Exception as e:
            logger.error(f"Failed to patch OpenAI: {e}")

    @staticmethod
    def _patch_anthropic():
        """Patch Anthropic client"""
        try:
            import importlib.util

            if importlib.util.find_spec("anthropic"):
                logger.warning("Anthropic auto-patching not implemented yet")
            else:
                logger.warning("Anthropic not available for patching")
        except ImportError:
            logger.warning("Anthropic not available for patching")

    @staticmethod
    def _patch_gemini():
        """Patch Google Gemini client"""
        try:
            import importlib.util

            if importlib.util.find_spec("google.generativeai"):
                logger.warning("Google Gemini not available for patching")
            else:
                logger.warning("Google Gemini not available for patching")
        except ImportError:
            logger.warning("Google Gemini not available for patching")


# =============================================================================
# PUBLIC API
# =============================================================================

# Global instances
monitor = Monitor()
_trainer_tester = None


def get_trainer_tester(db_path: str = "llmentary_training.db") -> TrainingTester:
    """Get the global trainer/tester instance"""
    global _trainer_tester
    if _trainer_tester is None:
        _trainer_tester = TrainingTester(db_path)
    return _trainer_tester


# Context managers and decorators for easy integration
@contextmanager
def training_mode(interactive: bool = True):
    """Context manager for training mode"""
    trainer = get_trainer_tester()
    old_mode = trainer.training_mode
    old_interactive = trainer.interactive

    trainer.training_mode = True
    trainer.interactive = interactive

    try:
        yield trainer
    finally:
        trainer.training_mode = old_mode
        trainer.interactive = old_interactive


def trainable(func):
    """Decorator to make a function trainable"""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        trainer = get_trainer_tester()
        if trainer.training_mode and len(args) > 0:
            # Assume first arg is the question and result is the answer
            question = str(args[0])
            answer = str(result)

            trainer.capture_interaction(
                question=question, answer=answer, model="unknown", provider="unknown"
            )

        return result

    return wrapper


def capture_interaction(
    question: str,
    answer: str,
    model: str = "unknown",
    provider: str = "unknown",
    **kwargs,
) -> Optional[int]:
    """Simple function to capture an interaction"""
    trainer = get_trainer_tester()
    return trainer.capture_interaction(question, answer, model, provider, **kwargs)


# Export main components
__all__ = [
    # Core classes
    "Monitor",
    "AutoInstrument",
    "TrainingTester",
    # Data structures
    "LLMCall",
    "TrainingExample",
    "TestResult",
    "DriftDetectionResult",
    # Storage
    "SQLiteStorage",
    "TrainingTestStorage",
    # Drift detection
    "AdvancedDriftDetector",
    "DriftDetectionConfig",
    "DriftType",
    "DriftSeverity",
    # Global instances
    "monitor",
    "get_trainer_tester",
    # Integration helpers
    "training_mode",
    "trainable",
    "capture_interaction",
]
