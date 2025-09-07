#!/usr/bin/env python3
"""
Demo script showing the new llmentary training/testing workflow

This demonstrates how to use llmentary for regression testing of LLM applications:
1. Training mode - save Q&A pairs you approve of
2. Testing mode - validate current responses against saved baselines
"""

import os
import sys
from training_tester import training_mode, capture_interaction, get_trainer_tester

def simulate_llm_call(question: str, model: str = "demo-model") -> str:
    """Simulate an LLM call - in real usage this would be your actual LLM"""
    
    # Simulated responses for demo purposes
    responses = {
        "What is our refund policy?": "We offer full refunds within 30 days of purchase.",
        "How do I reset my password?": "Click 'Forgot Password' on the login page and follow the email instructions.",
        "What are your business hours?": "We are open Monday-Friday 9 AM to 5 PM EST.",
        "How do I contact support?": "You can reach support at support@company.com or call 1-800-SUPPORT.",
    }
    
    # Sometimes return a slightly different response to demonstrate drift detection
    if "refund policy" in question.lower() and "v2" in model:
        return "We provide full refunds within 30 business days of purchase."  # Slightly different
    
    return responses.get(question, f"I don't have information about: {question}")

def demo_training_workflow():
    """Demonstrate the training workflow"""
    print("🎓 llmentary Training/Testing Demo")
    print("=" * 40)
    print()
    
    # Demo questions
    questions = [
        "What is our refund policy?",
        "How do I reset my password?", 
        "What are your business hours?",
        "How do I contact support?"
    ]
    
    print("1️⃣ TRAINING MODE")
    print("Collecting approved Q&A pairs...")
    print()
    
    # Training mode - collect approved examples
    with training_mode(interactive=False) as trainer:  # Non-interactive for demo
        for question in questions:
            answer = simulate_llm_call(question, "demo-model-v1")
            
            print(f"Q: {question}")
            print(f"A: {answer}")
            print("✅ Auto-saved (non-interactive mode)")
            print()
            
            # Manually capture since we're in non-interactive demo mode
            capture_interaction(
                question=question,
                answer=answer,
                model="demo-model-v1",
                provider="demo-provider",
                category="support"
            )
    
    print("2️⃣ TRAINING STATISTICS")
    trainer = get_trainer_tester()
    stats = trainer.get_training_stats()
    print(f"Total examples saved: {stats.get('total', 0)}")
    print(f"Categories: {stats.get('categories', {})}")
    print()
    
    print("3️⃣ REGRESSION TESTING")
    print("Testing current responses against saved baselines...")
    print()
    
    # Test current responses against saved examples
    examples = trainer.storage.get_examples()
    passed = 0
    failed = 0
    
    for example in examples:
        # Simulate current LLM response (might be different)
        current_answer = simulate_llm_call(example.question, "demo-model-v2")
        
        # Test against saved baseline
        result = trainer.test_example(example, current_answer)
        
        status = "✅ PASS" if result.matches else "❌ FAIL"
        print(f"{status} Q: {example.question[:50]}...")
        if not result.matches:
            print(f"     Expected hash: {result.expected_hash[:16]}...")
            print(f"     Actual hash:   {result.actual_hash[:16]}...")
            failed += 1
        else:
            passed += 1
        print()
    
    print("4️⃣ TEST RESULTS SUMMARY")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "No tests run")
    print()
    
    if failed > 0:
        print("🔍 DRIFT DETECTED!")
        print("Some responses have changed from your approved baselines.")
        print("Review the failed tests to determine if the changes are acceptable.")

if __name__ == "__main__":
    demo_training_workflow()