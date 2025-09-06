"""
llmentary - Privacy-first LLM monitoring and drift detection
"""

import hashlib
import json
import time
import sqlite3
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Optional structlog import with fallback
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    # Simple fallback logger
    class SimpleLogger:
        def info(self, msg, **kwargs):
            print(f"INFO: {msg}")
        def warning(self, msg, **kwargs):
            print(f"WARNING: {msg}")
        def error(self, msg, **kwargs):
            print(f"ERROR: {msg}")
    logger = SimpleLogger()

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

class Monitor:
    """Main monitoring class"""
    
    def __init__(self):
        self.config = {
            'store_raw_text': False,
            'drift_threshold': 0.85,
            'storage_backend': 'sqlite',
            'db_path': 'llmentary.db'
        }
        self.storage = SQLiteStorage(self.config['db_path'])
        self._initialize_db()
    
    def configure(self, **kwargs):
        """Configure the monitor"""
        self.config.update(kwargs)
        logger.info("Monitor configured", config=self.config)
    
    def _initialize_db(self):
        """Initialize the database"""
        self.storage.initialize()
    
    def record_call(self, input_text: str, output_text: str, 
                   model: str = "unknown", provider: str = "unknown", 
                   metadata: Dict[str, Any] = None):
        """Record an LLM call with comprehensive error handling"""
        try:
            metadata = metadata or {}
            
            # Validate inputs
            if not isinstance(input_text, str):
                input_text = str(input_text)
            if not isinstance(output_text, str):
                output_text = str(output_text)
            
            # Create hashes
            input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
            output_hash = hashlib.sha256(output_text.encode('utf-8')).hexdigest()
            
            # Create call record
            call = LLMCall(
                input_hash=input_hash,
                output_hash=output_hash,
                input_text=input_text if self.config['store_raw_text'] else None,
                output_text=output_text if self.config['store_raw_text'] else None,
                model=model,
                provider=provider,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Store the call
            self.storage.store_call(call)
            
            # Check for drift
            self._check_drift(call)
            
            return call
            
        except UnicodeDecodeError as e:
            logger.error(f"Unicode encoding error in record_call: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to record LLM call: {e}", 
                        provider=provider, model=model)
            return None
    
    def _check_drift(self, call: LLMCall):
        """Check if this call represents drift with error handling"""
        try:
            previous_calls = self.storage.get_calls_by_input_hash(call.input_hash)
            
            if len(previous_calls) > 1:
                # Check if output is different from previous calls
                previous_outputs = [c.output_hash for c in previous_calls[:-1]]
                if call.output_hash not in previous_outputs:
                    logger.warning("Drift detected!", 
                                 input_hash=call.input_hash[:8],
                                 new_output_hash=call.output_hash[:8],
                                 previous_outputs=[h[:8] for h in previous_outputs],
                                 model=call.model,
                                 provider=call.provider)
                                 
        except Exception as e:
            logger.error(f"Failed to check drift: {e}", 
                        input_hash=call.input_hash[:8] if call.input_hash else 'unknown')

class SQLiteStorage:
    """SQLite storage backend"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def initialize(self):
        """Initialize the database with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS llm_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_hash TEXT NOT NULL,
                        output_hash TEXT NOT NULL,
                        input_text TEXT,
                        output_text TEXT,
                        model TEXT,
                        provider TEXT,
                        timestamp TEXT,
                        metadata TEXT
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_input_hash ON llm_calls(input_hash)
                ''')
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise
    
    def store_call(self, call: LLMCall):
        """Store a call in the database with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO llm_calls 
                    (input_hash, output_hash, input_text, output_text, model, provider, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    call.input_hash,
                    call.output_hash,
                    call.input_text,
                    call.output_text,
                    call.model,
                    call.provider,
                    call.timestamp.isoformat(),
                    json.dumps(call.metadata, default=str)  # Handle non-serializable objects
                ))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store call in database: {e}", 
                        input_hash=call.input_hash[:8] if call.input_hash else 'unknown')
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"Data serialization error when storing call: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error storing call: {e}")
            raise
    
    def get_calls_by_input_hash(self, input_hash: str):
        """Get all calls with the same input hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM llm_calls WHERE input_hash = ? ORDER BY timestamp
            ''', (input_hash,))
            
            calls = []
            for row in cursor.fetchall():
                calls.append(LLMCall(
                    input_hash=row[1],
                    output_hash=row[2],
                    input_text=row[3],
                    output_text=row[4],
                    model=row[5],
                    provider=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8] or '{}')
                ))
            
            return calls

class AutoInstrument:
    """Auto-instrumentation for popular LLM libraries"""
    
    @staticmethod
    def auto_patch_all():
        """Automatically patch all supported LLM libraries"""
        # Try to patch OpenAI
        try:
            AutoInstrument.patch_openai()
        except ImportError:
            pass
        
        # Try to patch Anthropic
        try:
            AutoInstrument.patch_anthropic()
        except ImportError:
            pass
        
        # Try to patch Google Gemini
        try:
            AutoInstrument.patch_gemini()
        except ImportError:
            pass
    
    @staticmethod
    def patch_openai():
        """Patch OpenAI library for both v0.x and v1.0+ clients"""
        try:
            import openai
            
            # Check OpenAI version to determine which API to patch
            try:
                # Modern OpenAI client (v1.0+)
                if hasattr(openai, '__version__') and openai.__version__.startswith('1.'):
                    AutoInstrument._patch_openai_v1()
                else:
                    # Legacy OpenAI client (v0.x)
                    AutoInstrument._patch_openai_legacy()
            except Exception as e:
                # Fallback: try to detect by API structure
                if hasattr(openai, 'OpenAI'):
                    AutoInstrument._patch_openai_v1()
                elif hasattr(openai, 'ChatCompletion'):
                    AutoInstrument._patch_openai_legacy()
                else:
                    logger.error(f"Unable to determine OpenAI client version: {e}")
                    return
                    
            logger.info("OpenAI patched successfully")
            
        except ImportError:
            logger.warning("OpenAI not available for patching")
        except Exception as e:
            logger.error(f"Failed to patch OpenAI: {e}")
    
    @staticmethod
    def _patch_openai_v1():
        """Patch modern OpenAI client (v1.0+)"""
        import openai
        
        # Store original methods
        original_create = openai.OpenAI().chat.completions.create
        original_async_create = openai.AsyncOpenAI().chat.completions.create
        
        def monitored_create(self, *args, **kwargs):
            result = original_create.__get__(self, type(self))(*args, **kwargs)
            
            # Extract input and output
            messages = kwargs.get('messages', [])
            input_text = json.dumps(messages, default=str)
            output_text = result.choices[0].message.content
            
            # Record the call
            monitor.record_call(
                input_text=input_text,
                output_text=output_text,
                model=kwargs.get('model', 'unknown'),
                provider='openai',
                metadata={
                    'usage': getattr(result, 'usage', None).__dict__ if hasattr(result, 'usage') else {},
                    'response_id': getattr(result, 'id', None),
                    'created': getattr(result, 'created', None)
                }
            )
            
            return result
        
        async def monitored_async_create(self, *args, **kwargs):
            result = await original_async_create.__get__(self, type(self))(*args, **kwargs)
            
            # Extract input and output
            messages = kwargs.get('messages', [])
            input_text = json.dumps(messages, default=str)
            output_text = result.choices[0].message.content
            
            # Record the call
            monitor.record_call(
                input_text=input_text,
                output_text=output_text,
                model=kwargs.get('model', 'unknown'),
                provider='openai',
                metadata={
                    'usage': getattr(result, 'usage', None).__dict__ if hasattr(result, 'usage') else {},
                    'response_id': getattr(result, 'id', None),
                    'created': getattr(result, 'created', None),
                    'async': True
                }
            )
            
            return result
        
        # Patch the methods
        openai.OpenAI.chat.completions.create = monitored_create
        openai.AsyncOpenAI.chat.completions.create = monitored_async_create
    
    @staticmethod
    def _patch_openai_legacy():
        """Patch legacy OpenAI client (v0.x)"""
        import openai
        
        original_create = openai.ChatCompletion.create
        
        def monitored_create(*args, **kwargs):
            result = original_create(*args, **kwargs)
            
            # Extract input and output
            messages = kwargs.get('messages', [])
            input_text = json.dumps(messages, default=str)
            output_text = result.choices[0].message.content
            
            # Record the call
            monitor.record_call(
                input_text=input_text,
                output_text=output_text,
                model=kwargs.get('model', 'unknown'),
                provider='openai',
                metadata={
                    'usage': getattr(result, 'usage', None).__dict__ if hasattr(result, 'usage') else {},
                    'response_id': getattr(result, 'id', None),
                    'created': getattr(result, 'created', None)
                }
            )
            
            return result
        
        openai.ChatCompletion.create = monitored_create
    
    @staticmethod
    def patch_anthropic():
        """Patch Anthropic library for both sync and async clients"""
        try:
            import anthropic
            
            # Patch sync client
            original_create = anthropic.Anthropic().messages.create
            original_client_init = anthropic.Anthropic.__init__
            
            def monitored_create(self, *args, **kwargs):
                result = original_create.__get__(self, type(self))(*args, **kwargs)
                
                # Extract input and output
                messages = kwargs.get('messages', [])
                input_text = json.dumps(messages, default=str)
                output_text = result.content[0].text if result.content else ""
                
                # Record the call
                monitor.record_call(
                    input_text=input_text,
                    output_text=output_text,
                    model=kwargs.get('model', 'unknown'),
                    provider='anthropic',
                    metadata={
                        'usage': {
                            'input_tokens': getattr(result.usage, 'input_tokens', None) if hasattr(result, 'usage') else None,
                            'output_tokens': getattr(result.usage, 'output_tokens', None) if hasattr(result, 'usage') else None
                        },
                        'response_id': getattr(result, 'id', None),
                        'stop_reason': getattr(result, 'stop_reason', None)
                    }
                )
                
                return result
            
            # Patch async client
            original_async_create = anthropic.AsyncAnthropic().messages.create
            
            async def monitored_async_create(self, *args, **kwargs):
                result = await original_async_create.__get__(self, type(self))(*args, **kwargs)
                
                # Extract input and output
                messages = kwargs.get('messages', [])
                input_text = json.dumps(messages, default=str)
                output_text = result.content[0].text if result.content else ""
                
                # Record the call
                monitor.record_call(
                    input_text=input_text,
                    output_text=output_text,
                    model=kwargs.get('model', 'unknown'),
                    provider='anthropic',
                    metadata={
                        'usage': {
                            'input_tokens': getattr(result.usage, 'input_tokens', None) if hasattr(result, 'usage') else None,
                            'output_tokens': getattr(result.usage, 'output_tokens', None) if hasattr(result, 'usage') else None
                        },
                        'response_id': getattr(result, 'id', None),
                        'stop_reason': getattr(result, 'stop_reason', None),
                        'async': True
                    }
                )
                
                return result
            
            # Apply patches
            anthropic.Anthropic.messages.create = monitored_create
            anthropic.AsyncAnthropic.messages.create = monitored_async_create
            
            logger.info("Anthropic patched successfully")
            
        except ImportError:
            logger.warning("Anthropic not available for patching")
        except Exception as e:
            logger.error(f"Failed to patch Anthropic: {e}")
    
    @staticmethod
    def patch_gemini():
        """Patch Google Gemini library"""
        try:
            import google.generativeai as genai
            from google.generativeai import GenerativeModel
            
            # Store original method
            original_generate_content = GenerativeModel.generate_content
            
            def monitored_generate_content(self, *args, **kwargs):
                result = original_generate_content(self, *args, **kwargs)
                
                # Extract input and output
                if args:
                    input_text = str(args[0])  # First arg is usually the prompt
                else:
                    input_text = str(kwargs.get('contents', ''))
                
                output_text = result.text if hasattr(result, 'text') else str(result)
                
                # Record the call
                monitor.record_call(
                    input_text=input_text,
                    output_text=output_text,
                    model=getattr(self, 'model_name', 'unknown'),
                    provider='google',
                    metadata={
                        'usage': {
                            'prompt_token_count': getattr(result.usage_metadata, 'prompt_token_count', None) if hasattr(result, 'usage_metadata') else None,
                            'candidates_token_count': getattr(result.usage_metadata, 'candidates_token_count', None) if hasattr(result, 'usage_metadata') else None,
                            'total_token_count': getattr(result.usage_metadata, 'total_token_count', None) if hasattr(result, 'usage_metadata') else None
                        },
                        'finish_reason': getattr(result.candidates[0], 'finish_reason', None) if result.candidates else None,
                        'safety_ratings': [rating.__dict__ for rating in getattr(result.candidates[0], 'safety_ratings', [])] if result.candidates else []
                    }
                )
                
                return result
            
            # Apply patch
            GenerativeModel.generate_content = monitored_generate_content
            
            logger.info("Google Gemini patched successfully")
            
        except ImportError:
            logger.warning("Google Gemini not available for patching")
        except Exception as e:
            logger.error(f"Failed to patch Gemini: {e}")

def instrument(provider: str = "unknown", model: str = "unknown"):
    """Decorator to instrument a function"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Try to extract input/output from args and result
            input_text = str(args) if args else str(kwargs)
            output_text = str(result)
            
            monitor.record_call(
                input_text=input_text,
                output_text=output_text,
                model=model,
                provider=provider
            )
            
            return result
        return wrapper
    return decorator

# Global monitor instance
monitor = Monitor()

# Export main interfaces
__all__ = ['monitor', 'AutoInstrument', 'instrument']