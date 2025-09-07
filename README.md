# llmentary

Privacy-first LLM monitoring and regression testing toolkit. Detect model drift, collect training data, and ensure consistent LLM behavior with minimal code changes.

## 🎯 What is llmentary?

llmentary helps you:
- **Monitor LLM responses** with privacy-first hash-based drift detection
- **Build regression tests** by collecting approved question-answer pairs
- **Detect model drift** using advanced semantic similarity analysis
- **Maintain consistency** across model updates and deployments

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Integration (3 lines of code)

```python
from llmentary import monitor, AutoInstrument

# Enable monitoring
AutoInstrument.auto_patch_all()
monitor.configure(store_raw_text=False)  # Privacy-first

# Your LLM calls are now automatically monitored
# Drift detection happens transparently
```

### 3. Training Mode - Collect Approved Responses

```python
from llmentary import training_mode, capture_interaction

# Interactive training
with training_mode():
    response = your_llm_call("What is Python?")
    # You'll be prompted: "Save this Q&A pair? (y/n)"

# Or manual capture
response = your_llm_call("What is Python?") 
capture_interaction(
    question="What is Python?",
    answer=response,
    model="gpt-4",
    provider="openai"
)
```

### 4. Regression Testing

```python
from llmentary import get_trainer_tester

trainer = get_trainer_tester()
examples = trainer.storage.get_examples()

for example in examples:
    current_response = your_llm_call(example.question)
    result = trainer.test_example(example, current_response)
    
    if not result.matches:
        print(f"❌ DRIFT: {example.question}")
        print(f"Response changed from approved baseline")
```

## 📋 Complete Example

See `gemini_chat_demo.py` for a complete working example:

```bash
# Normal chat with monitoring
python gemini_chat_demo.py

# Training mode - save approved responses
python gemini_chat_demo.py --training

# Test mode - validate all saved responses
python gemini_chat_demo.py --test
```

## 🔐 Privacy & Security

- **Questions**: Stored in plaintext (needed for regression testing)
- **Answers**: Salted SHA-256 hashes only (privacy-preserving comparison)
- **Local Storage**: SQLite database in your project directory
- **User Control**: Nothing saved without explicit approval

## 🎛️ Configuration

```python
from llmentary import monitor

# Basic configuration
monitor.configure(
    store_raw_text=False,              # Privacy-first mode
    advanced_drift_detection=True,     # Enable semantic similarity
    drift_threshold=0.85               # Similarity threshold
)

# Advanced drift detection
from llmentary import DriftDetectionConfig, DriftType

config = DriftDetectionConfig(
    detection_mode=DriftType.HYBRID,   # EXACT, SEMANTIC, or HYBRID
    semantic_threshold=0.85,           # Semantic similarity threshold
    embedding_model="all-MiniLM-L6-v2" # Sentence transformer model
)
```

## 📊 Key Features

### 1. **Privacy-First Monitoring**
- Hash-based drift detection
- No sensitive data stored
- Local database only

### 2. **Training/Testing Workflow**
- Interactive approval of Q&A pairs
- Regression testing against baselines
- User-controlled data collection

### 3. **Advanced Drift Detection**
- Semantic similarity analysis
- Configurable thresholds
- Multiple detection modes (exact, semantic, hybrid)

### 4. **Easy Integration**
- Context managers and decorators
- Auto-instrumentation for major LLM providers
- <3 lines of code to get started

## 🛠️ Advanced Usage

### Context Manager Pattern
```python
from llmentary import training_mode

with training_mode(interactive=True):
    for question in questions:
        response = llm.ask(question)
        # Auto-prompted to save approved responses
```

### Decorator Pattern
```python
from llmentary import trainable

@trainable
def ask_support_question(question):
    return llm.ask(question)
```

### Integration with Web Frameworks

#### FastAPI
```python
from fastapi import FastAPI
from llmentary import capture_interaction

app = FastAPI()

@app.post("/ask")
async def ask_question(question: str):
    response = await llm.ask(question)
    capture_interaction(question, response, "gpt-4", "openai")
    return {"response": response}
```

## 📈 Workflow

1. **Monitor** - Add 3 lines to enable automatic drift detection
2. **Train** - Run your app with training mode to collect approved Q&A pairs  
3. **Test** - Validate current responses against saved baselines
4. **Iterate** - Refine your training data and catch regressions early

## 🧪 Testing Your Setup

```bash
# Test basic functionality
python -c "from llmentary import monitor; print('✅ llmentary imported successfully')"

# Run the demo
python gemini_chat_demo.py --help
```

## 📝 Requirements

- **Core**: Python 3.8+, numpy
- **Optional**: sentence-transformers (semantic analysis), click (better prompts)
- **LLM APIs**: Install as needed (openai, anthropic, google-generativeai)

## 🤝 Integration Examples

The toolkit works with any LLM provider:
- ✅ OpenAI GPT models
- ✅ Anthropic Claude
- ✅ Google Gemini
- ✅ Any custom LLM integration

---

**Get started in under 5 minutes. Privacy-first. User-controlled. Regression testing made simple.**