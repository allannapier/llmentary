# llmentary Integration Guide

## New Training/Testing Approach 🎯

llmentary has been redesigned as a **regression testing framework** for LLM applications. Instead of passive monitoring, you now have full control over what gets tested.

## Quick Start (3 Steps)

### 1. Training Mode - Collect Approved Q&A Pairs

```python
from llmentary.training_tester import training_mode, capture_interaction

# Option A: Interactive training mode
with training_mode():
    response = your_llm_call("What is our refund policy?")
    # User will be prompted: "Save this Q&A pair? (y/n)"

# Option B: Manual capture
response = your_llm_call("What is our refund policy?")
capture_interaction(
    question="What is our refund policy?",
    answer=response,
    model="gpt-4",
    provider="openai",
    category="support"
)
```

### 2. View Training Data

```bash
# See what you've collected
llmentary train stats
llmentary train list

# Add examples manually
llmentary add-example -q "What are your hours?" -m "gpt-4" -p "openai"
```

### 3. Regression Testing

```python
from llmentary.training_tester import get_trainer_tester

# Test current responses against saved baselines
trainer = get_trainer_tester()
examples = trainer.storage.get_examples(category="support")

for example in examples:
    current_response = your_llm_call(example.question)
    result = trainer.test_example(example, current_response)
    
    if not result.matches:
        print(f"❌ DRIFT: {example.question}")
        print(f"Response changed from approved baseline")
```

## Integration Patterns

### Pattern 1: Decorator (Simplest)
```python
from llmentary.training_tester import trainable

@trainable
def ask_support_question(question):
    return llm.ask(question)

# In training mode, you'll be prompted to save responses
```

### Pattern 2: Context Manager (Recommended)
```python
from llmentary.training_tester import training_mode

with training_mode():
    for question in support_questions:
        response = llm.ask(question)
        # Prompted to save each Q&A pair
```

### Pattern 3: Manual Control (Most Flexible)
```python
from llmentary.training_tester import capture_interaction

def process_user_question(question):
    response = llm.ask(question)
    
    # Only save if user is satisfied
    if user_approves_response(response):
        capture_interaction(
            question=question,
            answer=response,
            model="gpt-4",
            provider="openai",
            category="user_support"
        )
    
    return response
```

## CLI Commands

```bash
# Training
llmentary train start              # Interactive training mode
llmentary train list               # List saved examples
llmentary train stats              # Training statistics
llmentary add-example -q "..."     # Add example manually

# Testing  
llmentary test                     # Show test plan
llmentary test --category support  # Filter tests

# Legacy commands still available
llmentary status                   # System status
llmentary config --list            # Configuration
```

## Security Model

- ✅ **Questions**: Stored in plaintext (needed for regression testing)
- ✅ **Answers**: Hashed with salt (privacy-preserving comparison)
- ✅ **Local**: SQLite database in your project directory
- ✅ **User Control**: Nothing saved without explicit approval

## Example: Customer Support Bot

```python
# 1. Training Phase - Collect approved responses
from llmentary.training_tester import training_mode

support_questions = [
    "What is your refund policy?",
    "How do I reset my password?", 
    "What are your business hours?"
]

with training_mode():
    for question in support_questions:
        response = support_bot.ask(question)
        # User prompted to save if they approve the response

# 2. Later: Regression testing in CI/CD
trainer = get_trainer_tester()
examples = trainer.storage.get_examples(category="support")

failed_tests = []
for example in examples:
    current_response = support_bot.ask(example.question)
    result = trainer.test_example(example, current_response)
    
    if not result.matches:
        failed_tests.append({
            'question': example.question,
            'expected_hash': result.expected_hash,
            'actual_hash': result.actual_hash
        })

if failed_tests:
    print(f"❌ {len(failed_tests)} regression tests failed!")
    print("Support bot responses have changed from approved baselines")
else:
    print("✅ All regression tests passed!")
```

## Framework Integrations

### FastAPI
```python
from fastapi import FastAPI
from llmentary.training_tester import capture_interaction

app = FastAPI()

@app.post("/ask")
async def ask_question(question: str):
    response = await llm.ask(question)
    
    # Capture for regression testing
    capture_interaction(question, response, "gpt-4", "openai")
    
    return {"response": response}
```

### Flask
```python
from flask import Flask, request
from llmentary.training_tester import training_mode

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    
    with training_mode(interactive=False):  # Auto-save in production
        response = llm.ask(question)
    
    return {"response": response}
```

## CI/CD Integration

```yaml
# .github/workflows/llm-regression-tests.yml
name: LLM Regression Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install llmentary
        pip install -r requirements.txt
    
    - name: Run LLM regression tests
      run: |
        python -m pytest test_llm_regression.py
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Migration from Old Approach

If you were using the old passive monitoring approach:

```python
# OLD: Passive monitoring (remove this)
from llmentary import monitor, AutoInstrument
AutoInstrument.auto_patch_all()

# NEW: Training/testing workflow  
from llmentary.training_tester import training_mode

with training_mode():
    # Run your app, approve responses you want to test against
    pass
```

---

**Key Benefits:**
- 🎯 **Deliberate**: Only save Q&A pairs you explicitly approve
- 🔒 **Secure**: Questions plaintext, answers hashed
- 🧪 **Testable**: Built for regression testing, not just monitoring  
- 🚀 **Simple**: <3 lines of code to integrate
- 📊 **Actionable**: Clear pass/fail results, not just drift alerts