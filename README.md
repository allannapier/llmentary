# llmentary

llmentary is a privacy-first, automatic drift detection and monitoring tool for LLM-powered applications. It provides zero-effort integration, auto-instrumentation, and intelligent analytics to ensure your LLM responses remain consistent and reliable in production.

## Core Benefits Over Traditional Testing

- **Automatic Test Generation:** Learns from real usage patterns, no manual test writing required.
- **Zero-Effort Integration:** Just 3 lines of code to add monitoring:
  ```python
  from llmentary import monitor
  from llmentary.interceptors import AutoInstrument
  AutoInstrument.auto_patch_all()
  ```
- **Privacy-First Design:** Uses SHA-256 hashing by default, so sensitive data is never stored.
- **Intelligent Drift Detection:** Alerts when the same question gets different answers.

## How It Works

- Intercepts all LLM calls automatically (OpenAI, Anthropic, LangChain, etc.)
- Hashes inputs and outputs for privacy-preserving comparison
- Stores hashes and metadata in a local database
- Compares new responses against historical responses for the same input
- Alerts when drift is detected beyond configured thresholds

## Real-World Example

```python
# Your existing code doesn't change at all!
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is our refund policy?"}]
)
# Behind the scenes, the monitor:
# 1. Hashes "What is our refund policy?"
# 2. Checks if this question was asked before
# 3. Compares the new response with previous responses
# 4. Alerts if the answer has changed significantly
```

## Key Features

### 1. Auto-Instrumentation
- Automatically patches OpenAI, Anthropic, and LangChain libraries
- No need to modify existing code
- Works with async and sync functions

### 2. Smart Drift Detection
- **Hash-based:** Detects exact changes (no external dependencies)
- **Semantic:** Uses embeddings for similarity comparison (optional)
- Configurable thresholds for different use cases

### 3. Comprehensive Analytics

```bash
llmentary report        # See drift patterns
llmentary consistency   # Check consistency scores
llmentary inspect --input-hash abc123  # Deep dive into specific inputs
```

### 4. Flexible Alerting
- Console warnings (default)
- Slack notifications
- Email alerts
- Custom webhooks
- All configurable based on drift severity

### 5. Production-Ready Storage
- SQLite for easy start (no setup required)
- Extensible to Redis, PostgreSQL, MongoDB
- Batch writes for performance
- Automatic buffer flushing

## Real-World Use Cases
- **Model Updates:** Detect when model changes affect your app
- **Prompt Changes:** Ensure prompt modifications don't break functionality
- **Compliance:** Prove consistent responses for audit trails
- **Cost Optimization:** Identify when cheaper models give identical results
- **Quality Assurance:** Catch degradation before users notice

## Advanced Features

### Pattern Analysis
- Model: Which models are most consistent?
- Time: When drift occur most?
- Function: Which parts of your code are affected?

### Performance Optimization
- ~1-5ms overhead per call
- Async operations don't block your app
- Batched database writes
- Can be disabled/enabled at runtime

### Privacy Controls
```python
monitor.configure(
    store_raw_text=False,  # Only hashes (default)
    # or
    store_raw_text=True,   # Store actual text for debugging
)
```

## Migration Path
- Deploy alongside existing tests to build baseline
- Use drift detection to identify gaps in test coverage
- Generate test cases from real-world drift events
- Feed insights back into traditional test suites

## Comparison with Traditional Approach

| Traditional Testing      | Auto-Instrumentation         |
|-------------------------|------------------------------|
| Manual test case creation| Automatic learning from usage|
| Synthetic test data      | Real production inputs       |
| Requires maintenance     | Self-maintaining             |
| Run in CI/CD only        | Continuous monitoring        |
| Miss edge cases          | Catches all actual usage     |
| Hours to set up          | Minutes to deploy            |

## Why llmentary?
- Learns from reality rather than assumptions
- Requires no maintenance as it self-updates
- Catches issues immediately in production
- Preserves privacy through hashing
- Scales automatically with usage

llmentary creates a "fingerprint" of your LLM's behavior and alerts you whenever that fingerprint changes unexpectedly. It's like having automated regression tests that write themselves based on actual usage!