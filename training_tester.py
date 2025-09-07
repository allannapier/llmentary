"""
llmentary Training/Testing Framework

A deliberate regression testing system for LLM applications where developers:
1. Train by running their app and saving approved Q&A pairs
2. Test by validating current responses against saved baselines
3. Integrate with minimal code changes

This replaces the passive monitoring approach with user-controlled regression testing.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import click

try:
    from .llmentary import logger
except ImportError:
    from llmentary import logger

@dataclass
class TrainingExample:
    """A saved question-answer pair for regression testing"""
    id: Optional[int]
    question: str              # Stored in plaintext
    answer_hash: str          # Hashed for privacy
    model: str
    provider: str
    category: Optional[str]
    tags: List[str]
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'question': self.question,
            'answer_hash': self.answer_hash,
            'model': self.model,
            'provider': self.provider,
            'category': self.category,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
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

class TrainingTestStorage:
    """SQLite storage for training examples and test results"""
    
    def __init__(self, db_path: str = "llmentary_training.db"):
        self.db_path = Path(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the training database"""
        with sqlite3.connect(self.db_path) as conn:
            # Training examples table
            conn.execute('''
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
            ''')
            
            # Test results table
            conn.execute('''
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
            ''')
            
            # Indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON training_examples(category)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model ON training_examples(model)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_provider ON training_examples(provider)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_example ON test_results(example_id)')
            
            conn.commit()
            logger.info(f"Training database initialized at {self.db_path}")
    
    def save_example(self, example: TrainingExample) -> int:
        """Save a training example and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_examples 
                (question, answer_hash, model, provider, category, tags, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                example.question,
                example.answer_hash,
                example.model,
                example.provider,
                example.category,
                json.dumps(example.tags),
                example.created_at.isoformat(),
                json.dumps(example.metadata)
            ))
            
            example_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved training example {example_id}", 
                       question=example.question[:50] + "..." if len(example.question) > 50 else example.question)
            return example_id
    
    def get_examples(self, category: Optional[str] = None, 
                    model: Optional[str] = None,
                    provider: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> List[TrainingExample]:
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
                example_tags = json.loads(row[6] or '[]')
                
                # Filter by tags if specified
                if tags and not any(tag in example_tags for tag in tags):
                    continue
                    
                examples.append(TrainingExample(
                    id=row[0],
                    question=row[1],
                    answer_hash=row[2],
                    model=row[3],
                    provider=row[4],
                    category=row[5],
                    tags=example_tags,
                    created_at=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8] or '{}')
                ))
            
            return examples
    
    def save_test_result(self, result: TestResult) -> int:
        """Save a test result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO test_results 
                (example_id, expected_hash, actual_hash, matches, similarity_score, tested_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.example_id,
                result.expected_hash,
                result.actual_hash,
                result.matches,
                result.similarity_score,
                result.tested_at.isoformat(),
                json.dumps(result.metadata)
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            return result_id

class TrainingTester:
    """Main class for training/testing workflow"""
    
    def __init__(self, db_path: str = "llmentary_training.db"):
        self.storage = TrainingTestStorage(db_path)
        self.training_mode = False
        self.interactive = True
        
    def _hash_answer(self, answer: str, salt: str = "llmentary") -> str:
        """Create a salted hash of an answer"""
        salted = f"{salt}:{answer.strip()}"
        return hashlib.sha256(salted.encode('utf-8')).hexdigest()
    
    def _prompt_user_save(self, question: str, answer: str, model: str, provider: str) -> bool:
        """Prompt user to save this Q&A pair"""
        if not self.interactive:
            return True
            
        click.echo(f"\n📝 Training Question: {question}")
        click.echo(f"🤖 Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        click.echo(f"📊 Model: {model} | Provider: {provider}")
        
        return click.confirm("💾 Save this Q&A pair for regression testing?")
    
    def capture_interaction(self, question: str, answer: str, 
                          model: str = "unknown", provider: str = "unknown",
                          category: Optional[str] = None, 
                          tags: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          force_save: bool = False) -> Optional[int]:
        """Capture a Q&A interaction, optionally prompting user to save"""
        
        if self.training_mode:
            should_save = force_save or self._prompt_user_save(question, answer, model, provider)
            
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
                    metadata=metadata or {}
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
            metadata={}
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
            "newest": max(e.created_at for e in examples)
        }

# Global instance
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
                question=question,
                answer=answer,
                model="unknown",
                provider="unknown"
            )
        
        return result
    return wrapper

def capture_interaction(question: str, answer: str, 
                       model: str = "unknown", provider: str = "unknown",
                       **kwargs) -> Optional[int]:
    """Simple function to capture an interaction"""
    trainer = get_trainer_tester()
    return trainer.capture_interaction(question, answer, model, provider, **kwargs)