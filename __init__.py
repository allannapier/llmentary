"""
llmentary - Privacy-first LLM monitoring and drift detection

A zero-effort, automatic drift detection and monitoring tool for LLM-powered applications.
Provides intelligent analytics to ensure your LLM responses remain consistent and reliable.

Example usage:
    from llmentary import monitor, AutoInstrument
    
    # 3-line setup for automatic monitoring
    AutoInstrument.auto_patch_all()
    monitor.configure(store_raw_text=False)  # Privacy-first
    
    # Your existing LLM code works unchanged
    # All calls are automatically monitored and analyzed
"""

from .llmentary import (
    Monitor,
    AutoInstrument, 
    LLMCall,
    SQLiteStorage,
    monitor,
    instrument
)

# Version info
__version__ = "0.2.0"
__author__ = "llmentary team"

# Main public interface
__all__ = [
    # Core classes
    'Monitor',
    'AutoInstrument',
    'LLMCall', 
    'SQLiteStorage',
    
    # Main instances
    'monitor',
    
    # Decorators
    'instrument',
    
    # Metadata
    '__version__'
]